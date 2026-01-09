"""
Training Script for Forgery Detection - TensorFlow/Keras
=========================================================
Custom training loop with:
- Warmup + cosine annealing learning rate
- Mixed precision training
- Deep supervision
- Gradient clipping
- Early stopping
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm

from model import build_model
from losses import combined_loss, dice_loss, dice_coefficient, compute_deep_supervision_loss
from dataset import create_datasets, create_datasets_with_supplemental, create_tf_dataset


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_config(quick=True):
    """Get training configuration."""
    if quick:
        return {
            'target_size': 256,
            'epochs': 15,
            'batch_size': 4,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'val_split': 0.1,
            'patience': 5,
            'tta': 4,
            'threshold': 0.4,
            'use_amp': True,
            'warmup_epochs': 2,
        }
    else:
        return {
            'target_size': 384,
            'epochs': 40,
            'batch_size': 6,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'val_split': 0.1,
            'patience': 8,
            'tta': 8,
            'threshold': 0.4,
            'use_amp': True,
            'warmup_epochs': 3,
        }


def cosine_schedule_with_warmup(epoch, total_epochs, warmup_epochs, base_lr):
    """Compute learning rate with warmup and cosine annealing."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


@tf.function
def train_step(model, optimizer, images, masks, use_deep_supervision=True):
    """Single training step with gradient tape."""
    with tf.GradientTape() as tape:
        if use_deep_supervision:
            main_pred, ds4_pred, ds3_pred = model(images, training=True, return_deep=True)
            loss = compute_deep_supervision_loss(masks, main_pred, ds4_pred, ds3_pred)
        else:
            pred = model(images, training=True)
            loss = combined_loss(masks, pred)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Gradient clipping
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, main_pred if use_deep_supervision else pred


@tf.function
def val_step(model, images, masks, threshold=0.4):
    """Single validation step."""
    pred = model(images, training=False)
    loss = combined_loss(masks, pred)
    
    # Compute metrics
    pred_binary = tf.cast(pred > threshold, tf.float32)
    intersection = tf.reduce_sum(pred_binary * masks)
    pred_sum = tf.reduce_sum(pred_binary)
    gt_sum = tf.reduce_sum(masks)
    
    return loss, intersection, pred_sum, gt_sum


def train_model(
    config,
    train_masks_dir,
    train_images_dir,
    output_dir,
    supplemental_masks_dir=None,
    supplemental_images_dir=None,
    encoder_savedmodel_dir=None,
):
    """
    Main training function.
    
    Args:
        config: Training configuration dict
        train_masks_dir: Path to mask files
        train_images_dir: Path to training images
        output_dir: Output directory for model and logs
    
    Returns:
        Trained model and best F1 score
    """
    print("\n" + "=" * 60)
    print("FORGERY DETECTION - TENSORFLOW TRAINING")
    print("=" * 60)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    set_seed(42)
    
    # Mixed precision
    if config['use_amp']:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled (float16)")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        # Memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
        config['use_amp'] = False
    
    # Create datasets (split by case_id; union multi-masks)
    if supplemental_masks_dir or supplemental_images_dir:
        train_dataset, val_dataset = create_datasets_with_supplemental(
            train_masks_dir=train_masks_dir,
            train_images_dir=train_images_dir,
            supplemental_masks_dir=supplemental_masks_dir,
            supplemental_images_dir=supplemental_images_dir,
            target_size=config['target_size'],
            val_split=config['val_split'],
        )
    else:
        train_dataset, val_dataset = create_datasets(
            train_masks_dir, train_images_dir,
            target_size=config['target_size'],
            val_split=config['val_split']
        )
    
    train_loader = create_tf_dataset(train_dataset, config['batch_size'])
    val_loader = create_tf_dataset(val_dataset, config['batch_size'])
    
    # Build model
    model = build_model(
        input_shape=(config['target_size'], config['target_size'], 3),
        use_attention=True,
        freeze_encoder=config.get('freeze_encoder', False),
        encoder_type=config.get('encoder_type', 'resnet101v2_keras'),
        encoder_savedmodel_dir=encoder_savedmodel_dir or config.get('encoder_savedmodel_dir'),
        encoder_weights=config.get('encoder_weights', 'imagenet'),
    )
    
    total_params = model.count_params()
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_f1 = 0
    best_weights = None
    patience_cnt = 0
    
    steps_per_epoch = len(train_dataset) // config['batch_size']
    val_steps = len(val_dataset) // config['batch_size']
    
    print(f"\nConfig: {config['target_size']}px, {config['epochs']} epochs, batch={config['batch_size']}")
    print(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")
    print(f"\n{'Epoch':<6} {'Train':<10} {'Val':<10} {'F1':<10} {'LR':<12}")
    print("-" * 50)
    
    for epoch in range(config['epochs']):
        # Update learning rate
        lr = cosine_schedule_with_warmup(
            epoch, config['epochs'], config['warmup_epochs'], config['learning_rate']
        )
        optimizer.learning_rate.assign(lr)
        
        # Training
        train_loss = 0
        train_batches = 0
        
        for images, masks in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False, total=steps_per_epoch):
            loss, _ = train_step(model, optimizer, images, masks)
            train_loss += loss.numpy()
            train_batches += 1
        
        train_loss /= max(train_batches, 1)
        
        # Validation
        val_loss = 0
        val_batches = 0
        total_intersection = 0
        total_pred = 0
        total_gt = 0
        
        for images, masks in val_loader:
            loss, intersection, pred_sum, gt_sum = val_step(
                model, images, masks, config['threshold']
            )
            val_loss += loss.numpy()
            total_intersection += intersection.numpy()
            total_pred += pred_sum.numpy()
            total_gt += gt_sum.numpy()
            val_batches += 1
        
        val_loss /= max(val_batches, 1)
        
        # Compute F1
        precision = total_intersection / (total_pred + 1e-8)
        recall = total_intersection / (total_gt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) * 100
        
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()
            patience_cnt = 0
            marker = " *"
        else:
            patience_cnt += 1
        
        print(f"{epoch+1:<6} {train_loss:<10.4f} {val_loss:<10.4f} {f1:<9.2f}% {lr:<12.6f}{marker}")
        
        # Early stopping
        if patience_cnt >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    # Threshold sweep on validation set (pick best F1 / Dice threshold)
    thresholds = config.get("threshold_sweep")
    if thresholds is None:
        thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

    thr_stats = {float(t): {"intersection": 0.0, "pred_sum": 0.0, "gt_sum": 0.0} for t in thresholds}
    for images, masks in val_loader:
        pred = model(images, training=False)
        for t in thresholds:
            t = float(t)
            pred_binary = tf.cast(pred > t, tf.float32)
            thr_stats[t]["intersection"] += float(tf.reduce_sum(pred_binary * masks).numpy())
            thr_stats[t]["pred_sum"] += float(tf.reduce_sum(pred_binary).numpy())
            thr_stats[t]["gt_sum"] += float(tf.reduce_sum(masks).numpy())

    best_thr = None
    best_thr_f1 = -1.0
    for t in thresholds:
        t = float(t)
        inter = thr_stats[t]["intersection"]
        ps = thr_stats[t]["pred_sum"]
        gs = thr_stats[t]["gt_sum"]
        precision = inter / (ps + 1e-8)
        recall = inter / (gs + 1e-8)
        f1_t = 2 * precision * recall / (precision + recall + 1e-8) * 100
        if f1_t > best_thr_f1:
            best_thr_f1 = f1_t
            best_thr = t

    if best_thr is not None:
        print(f"\nBest threshold on val: {best_thr:.2f} (F1={best_thr_f1:.2f}%)")
        config["threshold"] = best_thr
    
    # Save model
    model_path = Path(output_dir) / 'model.keras'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Best F1: {best_f1:.2f}%")
    
    return model, best_f1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Forgery Detection Model')
    parser.add_argument('--train_masks_dir', default='/kaggle/input/forgery-detection/train_masks')
    parser.add_argument('--train_images_dir', default='/kaggle/input/forgery-detection/train_images')
    parser.add_argument('--output_dir', default='/kaggle/working')
    parser.add_argument('--quick', action='store_true', help='Quick training mode')
    parser.add_argument('--full', action='store_true', help='Full training mode')
    args = parser.parse_args()
    
    config = get_config(quick=args.quick or not args.full)
    
    model, best_f1 = train_model(
        config,
        args.train_masks_dir,
        args.train_images_dir,
        args.output_dir
    )
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE! Best F1: {best_f1:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
