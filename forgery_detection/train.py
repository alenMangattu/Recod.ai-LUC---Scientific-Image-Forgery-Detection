"""
Training Script for Forgery Detection
======================================
Full training pipeline with pretrained encoder.

Usage:
    python train.py --weights_path /kaggle/input/pretrained-weights/resnet101_imagenet.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import ResNet101UNet, ResNet50UNet, get_model
from dataset import ForgeryDataset, get_dataloaders
from losses import CombinedLoss, DeepSupervisionLoss
from predict import generate_submission


def train_epoch(model, dataloader, criterion, optimizer, device, use_deep_supervision=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    progress = tqdm(dataloader, desc="Training")
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward
        if use_deep_supervision:
            outputs = model(images, return_deep=True)
        else:
            outputs = model(images)
        
        # Loss
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics
    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_pred_pixels = 0
    total_gt_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Metrics
            pred_binary = (outputs > 0.5).float()
            total_correct += (pred_binary == masks).sum().item()
            total_pixels += masks.numel()
            total_intersection += (pred_binary * masks).sum().item()
            total_pred_pixels += pred_binary.sum().item()
            total_gt_pixels += masks.sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_pixels * 100
    
    # Precision, Recall, F1
    precision = total_intersection / (total_pred_pixels + 1e-8) * 100
    recall = total_intersection / (total_gt_pixels + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(config):
    """Main training function."""
    print("=" * 60)
    print("FORGERY DETECTION TRAINING")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        train_masks_dir=config['train_masks_dir'],
        train_images_dir=config['train_images_dir'],
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        val_split=config['val_split'],
        num_workers=config['num_workers']
    )
    
    # Model
    print("\nCreating model...")
    model = get_model(
        model_name=config['model'],
        weights_path=config['weights_path'],
        freeze_encoder=config.get('freeze_encoder', False)
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss
    if config.get('use_deep_supervision', False):
        criterion = DeepSupervisionLoss()
    else:
        criterion = CombinedLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler - CosineAnnealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config['epochs'] // 3,  # Restart every 1/3 of training
        T_mult=1,
        eta_min=config['learning_rate'] / 100
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print(f"Training for {config['epochs']} epochs")
    print(f"Batch size: {config['batch_size']}, LR: {config['learning_rate']}")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_deep_supervision=config.get('use_deep_supervision', False)
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f} | Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}% | F1: {val_metrics['f1']:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model (by F1 score, not loss)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✓ New best F1: {best_val_f1:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (F1: {best_val_f1:.2f}%, Loss: {best_val_loss:.6f})")
    
    # Save model
    if config.get('save_path'):
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_val_f1': best_val_f1,
            'history': history
        }, config['save_path'])
        print(f"✓ Model saved to: {config['save_path']}")
    
    # Plot training history
    if config.get('plot', True):
        plot_history(history, config.get('plot_path', 'training_history.png'))
    
    return model, history


def plot_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision/Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].set_title('Precision & Recall', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1
    axes[1, 1].plot(history['val_f1'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score (%)')
    axes[1, 1].set_title('Validation F1 Score', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train Forgery Detection Model')
    
    # Paths
    parser.add_argument('--train_masks_dir', type=str, default='train_masks',
                        help='Directory with training masks')
    parser.add_argument('--train_images_dir', type=str, default='train_images',
                        help='Directory with training images')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--save_path', type=str, default='forgery_model.pth',
                        help='Path to save trained model')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet101',
                        choices=['resnet101', 'resnet50'])
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Options
    parser.add_argument('--use_deep_supervision', action='store_true')
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    config = vars(args)
    config['plot'] = not args.no_plot
    
    # Train
    model, history = train(config)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
