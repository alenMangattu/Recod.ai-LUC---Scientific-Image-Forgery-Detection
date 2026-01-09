"""
Scientific Image Forgery Detection - U-Net with ResNet101 Encoder
Kaggle Competition: Recod.ai/LUC - Scientific Image Forgery Detection
"""

import os
import sys
import signal
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.applications as applications
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# OFFICIAL COMPETITION METRIC (F1 Score)
# ============================================================================

def calculate_f1_score(pred_mask, gt_mask):
    """Calculate F1 score between predicted and ground truth masks"""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0

class F1ScoreMetric(keras.metrics.Metric):
    """Custom Keras metric for F1 Score"""
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + keras.backend.epsilon())
        return f1
    
    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

# ============================================================================
# CONFIGURATION - Auto-detect Kaggle vs Local
# ============================================================================
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🔵 Running on Kaggle")
    KAGGLE_INPUT = Path('/kaggle/input')
    KAGGLE_WORKING = Path('/kaggle/working')
    COMPETITION_DATA = KAGGLE_INPUT / 'recodai-luc-scientific-image-forgery-detection'
    
    TRAIN_IMAGES_DIR = COMPETITION_DATA / 'train_images'
    TRAIN_MASKS_DIR = COMPETITION_DATA / 'train_masks'
    TEST_IMAGES_DIR = COMPETITION_DATA / 'test_images'
    SUPPLEMENTAL_IMAGES_DIR = COMPETITION_DATA / 'supplemental_images'
    SUPPLEMENTAL_MASKS_DIR = COMPETITION_DATA / 'supplemental_masks'
    
    # Local ResNet101 weights (no internet access)
    RESNET_WEIGHTS_PATH = KAGGLE_INPUT / 'resnet101-weights-tf-dim-ordering-tf-kernels-notop/tensorflow2/default/1/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    OUTPUT_PATH = KAGGLE_WORKING / 'submission.csv'
else:
    print("🟢 Running locally")
TRAIN_IMAGES_DIR = Path('train_images')
TRAIN_MASKS_DIR = Path('train_masks')
    TEST_IMAGES_DIR = Path('test_images')
    SUPPLEMENTAL_IMAGES_DIR = Path('supplemental_images')
    SUPPLEMENTAL_MASKS_DIR = Path('supplemental_masks')
    
    # Use ImageNet weights when running locally (has internet)
    RESNET_WEIGHTS_PATH = 'imagenet'
    
    OUTPUT_PATH = Path('submission.csv')

IMG_SIZE = (256, 256)
BATCH_SIZE = 32  # Increased for faster training
EPOCHS = 50  # More epochs for proper training
LEARNING_RATE = 1e-4
TRAIN_SIZE = None  # Use ALL data (set to 100 for testing)
VAL_SIZE = 200  # More validation samples
HOLDOUT_SIZE = 100  # More holdout samples for better evaluation
SPLIT_SEED = 1337  # deterministic shuffle for splits
AUGMENT = True  # Enable data augmentation

# Training control
# - Set EARLY_STOP_PATIENCE=None to always run full EPOCHS
# - If enabled, we monitor val_f1_score (competition metric)
EARLY_STOP_PATIENCE = None if IS_KAGGLE else 20
EARLY_STOP_MIN_DELTA = 0.0

# Optional: reduce TF logging noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================================
# DATA LOADING
# ============================================================================

def find_image(mask_name):
    """Find corresponding image for a mask (checks train and supplemental dirs)"""
    name = mask_name.stem
    
    # Check train_images first
    for folder in ['forged', 'authentic']:
        path = TRAIN_IMAGES_DIR / folder / f'{name}.png'
        if path.exists():
            return path, folder
    
    # Check supplemental_images if on Kaggle
    if IS_KAGGLE and SUPPLEMENTAL_IMAGES_DIR.exists():
        for folder in ['forged', 'authentic']:
            path = SUPPLEMENTAL_IMAGES_DIR / folder / f'{name}.png'
            if path.exists():
                return path, folder
    
    return None, None

def getdata(index, mask_files):
    """Load image and mask for given index"""
    mask_path = mask_files[index]
    image_path, label = find_image(mask_path)

    # Always force RGB (3 channels)
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)  # (H,W,3)

    # Load mask and handle different shapes
    mask = np.load(mask_path)
    mask = np.squeeze(mask)

    if mask.ndim == 3:
        # Handle (C,H,W) like (2,H,W) or (3,H,W) - take max across channels
        if mask.shape[0] in (2, 3):
            mask = mask.max(axis=0)
        # Handle (H,W,C)
        elif mask.shape[-1] in (2, 3):
            mask = mask.max(axis=-1)
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[..., 0]

    mask = mask.astype(np.float32)
    if mask.max() > 1:
        mask /= mask.max()

    return img, mask

def augment_image_and_mask(img, mask):
    """Apply data augmentation to image and mask together"""
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    # Random vertical flip
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    
    # Random 90-degree rotations
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    
    # Random brightness (image only)
    img = tf.image.random_brightness(img, max_delta=0.2)
    
    # Random contrast (image only)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # Random saturation (image only)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    
    # Random hue (image only, small range)
    img = tf.image.random_hue(img, max_delta=0.05)
    
    # Clip image values to valid range
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask


def create_dataset(indices, mask_files, batch_size=4, img_size=(256, 256), training=False):
    """Create TensorFlow dataset from indices"""
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(indices, tf.int32))
    
    if training:
        ds = ds.shuffle(len(indices), reshuffle_each_iteration=True)

    def _py_load(i):
        i = int(i)
        img, mask = getdata(i, mask_files)
        return img, mask

    def _map(i):
        img, mask = tf.py_function(_py_load, [i], [tf.uint8, tf.float32])
        img.set_shape([None, None, 3])
        mask.set_shape([None, None])

        img = tf.image.resize(tf.cast(img, tf.float32), img_size) / 255.0
        mask = tf.image.resize(mask[..., None], img_size, method="nearest")
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        return img, mask
    
    def _augment(img, mask):
        return augment_image_and_mask(img, mask)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    
    # Apply augmentation only during training
    if training and AUGMENT:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.cache().repeat()
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_unet(input_shape=(256, 256, 3), num_classes=1):
    """Build U-Net with ResNet101 encoder"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (ResNet101 backbone)
    # Use local weights on Kaggle (no internet), ImageNet otherwise
    if IS_KAGGLE and RESNET_WEIGHTS_PATH.exists():
        print(f"Loading ResNet101 weights from: {RESNET_WEIGHTS_PATH}")
        base_model = applications.ResNet101(
            include_top=False,
            weights=None,  # Don't download
            input_tensor=inputs
        )
        base_model.load_weights(str(RESNET_WEIGHTS_PATH))
    else:
        print("Loading ResNet101 with ImageNet weights")
    base_model = applications.ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Get encoder layers at different resolutions
    encoder_outputs = [
        base_model.get_layer('conv1_relu').output,           # 1/2
        base_model.get_layer('conv2_block3_out').output,      # 1/4
        base_model.get_layer('conv3_block4_out').output,      # 1/8
        base_model.get_layer('conv4_block23_out').output,     # 1/16
        base_model.get_layer('conv5_block3_out').output,      # 1/32 (bottleneck)
    ]
    
    # Decoder - use UpSampling2D instead of Conv2DTranspose to avoid checkerboard artifacts
    # Added Dropout for regularization to reduce overfitting
    x = encoder_outputs[-1]
    
    def upsample_block(x, skip, filters, dropout_rate=0.3):
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout for regularization
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout for regularization
    return x
    
    x = upsample_block(x, encoder_outputs[3], 512, dropout_rate=0.4)
    x = upsample_block(x, encoder_outputs[2], 256, dropout_rate=0.3)
    x = upsample_block(x, encoder_outputs[1], 128, dropout_rate=0.2)
    x = upsample_block(x, encoder_outputs[0], 64, dropout_rate=0.1)
    
    # Final upsample to match input size
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)  # Light dropout before final layer
    
    # Final output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs, name='unet_resnet101')
    return model

# ============================================================================
# TRAINING
# ============================================================================

def setup_runtime():
    """Print device info and enable a few safe performance knobs."""
    print("\nTensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs detected: {gpus}")
        # Memory growth helps on CUDA; on Apple Metal it may no-op or raise.
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth (ok):", repr(e))
        print("Logical GPUs:", tf.config.list_logical_devices("GPU"))
    else:
        print("No GPU detected. Training will run on CPU.")

    # Optional mixed precision (enable by setting env MIXED_PRECISION=1)
    if os.environ.get("MIXED_PRECISION", "0") == "1":
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled: mixed_float16")
        except Exception as e:
            print("Could not enable mixed precision (skipping):", repr(e))


class GracefulInterruptCallback(keras.callbacks.Callback):
    """
    First Ctrl-C: request a clean stop at the end of the current batch.
    Second Ctrl-C: force exit.
    """

    def __init__(self):
        super().__init__()
        self._stop_requested = False
        self._old_handler = None

    def on_train_begin(self, logs=None):
        def _handler(sig, frame):
            if self._stop_requested:
                print("\nSecond Ctrl-C detected -> force exit.")
                os._exit(130)
            self._stop_requested = True
            print("\nCtrl-C detected -> stopping after this batch (saving checkpoint). Press again to force.")

        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handler)

    def on_train_batch_end(self, batch, logs=None):
        if self._stop_requested:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)


class TimeEstimateCallback(keras.callbacks.Callback):
    """Minimal training progress callback"""
    
    def __init__(self, total_epochs, total_steps=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start_time = None
    
    def _format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining = avg_time * (self.total_epochs - epoch - 1)
        elapsed = time.time() - self.start_time
        
        loss = logs.get('loss', 0) if logs else 0
        val_iou = logs.get('val_iou', 0) if logs else 0
        
        print(f"Epoch {epoch+1}/{self.total_epochs} | Loss: {loss:.4f} | Val IoU: {val_iou:.4f} | Time: {self._format_time(epoch_time)} | ETA: {self._format_time(remaining)}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"Training complete: {self._format_time(total_time)}")


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky Loss - handles class imbalance better
    alpha > beta: penalize false negatives more (encourages detecting forgeries)
    gamma < 1: focal component to focus on hard examples
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.pow(1 - tversky, gamma)

def train_model(model, train_ds, val_ds, train_steps, val_steps, epochs=10):
    """Train the model"""
    
    # Compile model with Focal Tversky loss (handles class imbalance)
    # Now monitoring F1 score as per competition metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_tversky_loss,
        metrics=[
            'accuracy', 
            keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'),
            F1ScoreMetric(threshold=0.5, name='f1_score')
        ]
    )
    
    # Callbacks - monitoring val_f1_score (competition metric)
    model_save_path = '/kaggle/working/best_model.keras' if IS_KAGGLE else 'best_model.keras'
    callbacks = [
        GracefulInterruptCallback(),
        keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True,
            monitor='val_f1_score',  # Monitor F1 (competition metric)
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',  # LR reduction based on F1
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Early stopping: disabled for Kaggle (train full epochs), enabled locally
    if not IS_KAGGLE and EARLY_STOP_PATIENCE is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_f1_score',
                mode='max',
                patience=int(EARLY_STOP_PATIENCE),
                min_delta=float(EARLY_STOP_MIN_DELTA),
                restore_best_weights=True,
                verbose=1,
            )
        )
        print(f"EarlyStopping: ON (patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA})")
    else:
        if IS_KAGGLE:
            print("EarlyStopping: OFF (Kaggle mode - will train for all epochs)")
        else:
            print("EarlyStopping: OFF (will train for all epochs)")
    
    try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    return history
    except KeyboardInterrupt:
        # Should be rare because GracefulInterruptCallback stops training cleanly,
        # but keep as a final safety net.
        print("\nKeyboardInterrupt -> saving interrupted model to interrupted_model.keras")
        try:
            model.save("interrupted_model.keras")
        except Exception as e:
            print("Could not save interrupted model:", repr(e))
        raise

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")

def visualize_predictions(model, mask_files, indices, num_samples=3):
    """Visualize model predictions"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, idx in enumerate(indices[:num_samples]):
        img, true_mask = getdata(idx, mask_files)
        original_shape = img.shape[:2]
        
        # Preprocess
        img_resized = tf.image.resize(img, IMG_SIZE)
        img_tensor = tf.expand_dims(img_resized / 255.0, 0)
        
        # Predict
        pred = model.predict(img_tensor, verbose=0)
        pred_mask = pred[0, :, :, 0]
        
        # Resize back
        pred_mask_resized = tf.image.resize(
            pred_mask[..., None], 
            original_shape, 
            method='nearest'
        ).numpy()[:, :, 0]
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image (idx {idx})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask, cmap='hot')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask_resized, cmap='hot')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions saved to predictions.png")

# ============================================================================
# SUBMISSION
# ============================================================================

def rle_encode(mask):
    """
    Official competition RLE encoding format.
    Returns JSON string like "[123, 4, 56, 7]" or "authentic" if no forgery.
    """
    # Transpose and flatten (matches official implementation)
    pixels = mask.T.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    # If no forgery pixels, return "authentic"
    if pixels.sum() == 0:
        return "authentic"
    
    # Find all foreground pixels
    dots = np.where(pixels == 1)[0]
    
    # Build RLE pairs using official algorithm
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            # Start new run
            run_lengths.extend([int(b + 1), 0])  # +1 for 1-indexing
        run_lengths[-1] += 1  # Increment length
        prev = b
    
    # Format as JSON string (official format)
    return json.dumps(run_lengths)

def predict_test_images(model, test_image_paths, img_size=(256, 256)):
    """Generate predictions for test images (unseen data)"""
    submissions = []
    
    print(f"\nGenerating predictions for {len(test_image_paths)} test images...")
    
    for i, img_path in enumerate(test_image_paths):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(test_image_paths)}...")
        
        try:
            # Load test image (no mask available)
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            original_shape = img.shape[:2]
            
            # Get case_id from filename (without extension)
            case_id = img_path.stem
            
            # Preprocess
            img_resized = tf.image.resize(img, img_size)
            img_tensor = tf.expand_dims(img_resized / 255.0, 0)
            
            # Predict
            pred = model.predict(img_tensor, verbose=0)
            pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # Resize back to original
            pred_mask_resized = tf.image.resize(
                pred_mask[..., None], 
                original_shape, 
                method='nearest'
            ).numpy()[:, :, 0]
            
            # Create submission entry (rle_encode returns "authentic" if no forgery)
            annotation = rle_encode(pred_mask_resized)
            submissions.append({"case_id": case_id, "annotation": annotation})
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            case_id = img_path.stem
            submissions.append({"case_id": case_id, "annotation": "authentic"})
    
    return submissions

def evaluate_validation(model, mask_files, val_indices, img_size=(256, 256), verbose=True):
    """Evaluate model on validation set with comprehensive diagnostics"""
    all_preds = []
    all_true = []
    forgery_correct = 0
    forgery_total = 0
    authentic_correct = 0
    authentic_total = 0
    
    # Diagnostics: track raw prediction values
    raw_pred_values = []
    per_image_stats = []
    f1_scores = []
    
    for idx in val_indices:
        try:
            img, true_mask = getdata(idx, mask_files)
            img_resized = tf.image.resize(img, img_size)
            img_tensor = tf.expand_dims(img_resized / 255.0, 0)
            pred = model.predict(img_tensor, verbose=0)
            pred_mask = pred[0, :, :, 0]
            pred_mask_resized = tf.image.resize(pred_mask[..., None], img.shape[:2], method='nearest').numpy()[:, :, 0]
            
            # Store raw values for diagnostics
            raw_pred_values.extend(pred_mask_resized.flatten())
            
            pred_binary = (pred_mask_resized > 0.5).astype(np.uint8)
            true_binary = (true_mask > 0.5).astype(np.uint8)
            
            # Per-image F1 score
            img_f1 = calculate_f1_score(pred_binary, true_binary)
            f1_scores.append(img_f1)
            
            # Per-image stats
            per_image_stats.append({
                'pred_min': pred_mask_resized.min(),
                'pred_max': pred_mask_resized.max(),
                'pred_mean': pred_mask_resized.mean(),
                'pred_pos_pixels': pred_binary.sum(),
                'true_pos_pixels': true_binary.sum(),
                'f1': img_f1
            })
            
            all_preds.extend(pred_binary.flatten())
            all_true.extend(true_binary.flatten())
            
            # Classification stats
            true_is_forgery = true_binary.sum() > 0
            pred_is_forgery = pred_binary.sum() > 0
            
            if true_is_forgery:
                forgery_total += 1
                if pred_is_forgery:
                    forgery_correct += 1
            else:
                authentic_total += 1
                if not pred_is_forgery:
                    authentic_correct += 1
        except Exception as e:
            if verbose:
                print(f"Error evaluating idx {idx}: {e}")
            continue
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    raw_pred_array = np.array(raw_pred_values)
    
    # Calculate metrics
    tp = np.sum((all_preds == 1) & (all_true == 1))
    fp = np.sum((all_preds == 1) & (all_true == 0))
    fn = np.sum((all_preds == 0) & (all_true == 1))
    tn = np.sum((all_preds == 0) & (all_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    intersection = np.logical_and(all_preds, all_true).sum()
    union = np.logical_or(all_preds, all_true).sum()
    iou = intersection / union if union > 0 else 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Main metrics
        print(f"\n📊 Metrics:")
        print(f"   F1 Score: {f1:.4f} (Competition Metric)")
        print(f"   IoU: {iou:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        
        # Classification accuracy
        class_acc = (forgery_correct + authentic_correct) / max(1, forgery_total + authentic_total)
        print(f"\n📋 Classification:")
        print(f"   Forgery Detection: {forgery_correct}/{forgery_total} ({100*forgery_correct/max(1,forgery_total):.1f}%)")
        print(f"   Authentic Detection: {authentic_correct}/{authentic_total} ({100*authentic_correct/max(1,authentic_total):.1f}%)")
        print(f"   Overall Classification Acc: {class_acc:.4f}")
        
        # Prediction diagnostics
        print(f"\n🔍 Prediction Diagnostics (before thresholding):")
        print(f"   Min: {raw_pred_array.min():.4f}")
        print(f"   Max: {raw_pred_array.max():.4f}")
        print(f"   Mean: {raw_pred_array.mean():.4f}")
        print(f"   Median: {np.median(raw_pred_array):.4f}")
        
        # Distribution
        print(f"\n📈 Prediction Distribution:")
        print(f"   < 0.1: {(raw_pred_array < 0.1).mean()*100:.1f}%")
        print(f"   0.1-0.3: {((raw_pred_array >= 0.1) & (raw_pred_array < 0.3)).mean()*100:.1f}%")
        print(f"   0.3-0.5: {((raw_pred_array >= 0.3) & (raw_pred_array < 0.5)).mean()*100:.1f}%")
        print(f"   0.5-0.7: {((raw_pred_array >= 0.5) & (raw_pred_array < 0.7)).mean()*100:.1f}%")
        print(f"   > 0.7: {(raw_pred_array > 0.7).mean()*100:.1f}%")
        
        # Issue detection
        print(f"\n⚠️  Diagnostic Notes:")
        if raw_pred_array.max() < 0.5:
            print(f"   - Model predictions never exceed 0.5 (all zeros after threshold)")
        if raw_pred_array.mean() > 0.7:
            print(f"   - Model predicting too many positives (mean > 0.7)")
        if f1 < 0.01 and forgery_total > 0:
            print(f"   - Very low F1 despite having forged images in validation")
        if forgery_total == 0:
            print(f"   - No forged images in validation set!")
        
        # Sample image stats
        print(f"\n📷 Sample Images (first 5):")
        for i, stat in enumerate(per_image_stats[:5]):
            print(f"   Image {i+1}: pred range [{stat['pred_min']:.3f}, {stat['pred_max']:.3f}], "
                  f"pred/true pixels: {stat['pred_pos_pixels']}/{stat['true_pos_pixels']}, F1: {stat['f1']:.4f}")
        
        print(f"{'='*60}\n")
    
    return {
        'f1': f1,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'forgery_acc': forgery_correct / max(1, forgery_total),
        'authentic_acc': authentic_correct / max(1, authentic_total)
    }

# ============================================================================
# MAIN
# ============================================================================

def validate_data(mask_files, sample_size=100):
    """Validate data and print statistics"""
    forged_count = 0
    authentic_count = 0
    bad_samples = []
    
    indices = np.random.choice(len(mask_files), min(sample_size, len(mask_files)), replace=False)
    
    for i in indices:
        try:
            img, mask = getdata(i, mask_files)
            mask_path = mask_files[i]
            img_path, label = find_image(mask_path)
            
            if label == 'forged':
                forged_count += 1
            else:
                authentic_count += 1
            
            if img.shape[:2] != mask.shape[:2]:
                bad_samples.append({'idx': i, 'reason': 'shape_mismatch'})
                
        except Exception as e:
            bad_samples.append({'idx': i, 'reason': str(e)})
    
    print(f"Data: {forged_count} forged, {authentic_count} authentic, {len(bad_samples)} bad samples")
    return bad_samples


def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("Scientific Image Forgery Detection")
    print("Kaggle Competition: Recod.ai/LUC")
    print("="*60)
    
    setup_runtime()
    
    # Load training data
    print("\n📁 Loading data...")
    mask_files = list(TRAIN_MASKS_DIR.glob('*.npy'))
    print(f"   Train masks: {len(mask_files)}")
    
    # Also load supplemental data if available (Kaggle)
    supplemental_masks = []
    if IS_KAGGLE and SUPPLEMENTAL_MASKS_DIR.exists():
        supplemental_masks = list(SUPPLEMENTAL_MASKS_DIR.glob('*.npy'))
        print(f"   Supplemental masks: {len(supplemental_masks)}")
        mask_files = mask_files + supplemental_masks
        print(f"   Total masks: {len(mask_files)}")
    
    validate_data(mask_files, sample_size=200)
    
    # Load test images
    test_image_paths = sorted(TEST_IMAGES_DIR.glob('*.png'))
    print(f"   Test images: {len(test_image_paths)}")
    
    total_n = len(mask_files) if TRAIN_SIZE is None else min(TRAIN_SIZE, len(mask_files))
    if total_n < (VAL_SIZE + HOLDOUT_SIZE + 1):
        raise ValueError(f"Not enough samples: {total_n} < {VAL_SIZE + HOLDOUT_SIZE + 1}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(total_n)
    holdout_indices = perm[:HOLDOUT_SIZE].tolist()
    val_indices = perm[HOLDOUT_SIZE:HOLDOUT_SIZE + VAL_SIZE].tolist()
    train_indices = perm[HOLDOUT_SIZE + VAL_SIZE:].tolist()

    # Kaggle mode: train on the entire training set (do not hold out validation/holdout),
    # but still compute validation metrics for monitoring.
    if IS_KAGGLE:
        train_indices = perm.tolist()
        holdout_indices = []

    print(f"\n📊 Data splits:")
    if IS_KAGGLE:
        print(f"   Train: {len(train_indices)} (ALL data)")
        print(f"   Val: {len(val_indices)} (monitoring only; not excluded from training)")
    else:
        print(f"   Train: {len(train_indices)} | Val: {len(val_indices)} | Holdout: {len(holdout_indices)}")
    
    # Create datasets
    print("\n🔧 Creating datasets...")
    train_ds = create_dataset(train_indices, mask_files, batch_size=BATCH_SIZE, img_size=IMG_SIZE, training=True)
    val_ds = create_dataset(val_indices, mask_files, batch_size=BATCH_SIZE, img_size=IMG_SIZE, training=False)
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_unet(input_shape=(*IMG_SIZE, 3), num_classes=1)
    model.summary(print_fn=lambda x: None)  # Suppress verbose summary
    print(f"   Model parameters: {model.count_params():,}")
    
    # Train
    train_steps = len(train_indices) // BATCH_SIZE
    val_steps = len(val_indices) // BATCH_SIZE
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"   Steps/epoch: {train_steps} (train), {val_steps} (val)")
    print("-"*60)
    
    try:
        history = train_model(model, train_ds, val_ds, train_steps, val_steps, epochs=EPOCHS)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return
    
    # Visualizations (skip on Kaggle to save time)
    if not IS_KAGGLE:
        plot_training_history(history)
    visualize_predictions(model, mask_files, val_indices, num_samples=3)
    
    # Evaluate
    print("\n📈 Evaluating on validation set...")
    evaluate_validation(model, mask_files, val_indices, img_size=IMG_SIZE)
    
    if holdout_indices:
        print("\n📈 Evaluating on holdout set...")
        evaluate_validation(model, mask_files, holdout_indices, img_size=IMG_SIZE)

    # Generate submission for test images
    print("\n"+"="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    
    if len(test_image_paths) > 0:
        print(f"\n🔮 Predicting {len(test_image_paths)} test images...")
        submissions = predict_test_images(model, test_image_paths, img_size=IMG_SIZE)
    
        # Create DataFrame with correct column order
    df = pd.DataFrame(submissions)
        
        # Ensure columns are in correct order: case_id, annotation
        df = df[['case_id', 'annotation']]
        
        # Save to output path
        df.to_csv(str(OUTPUT_PATH), index=False)
        
        # Statistics
        authentic = sum(1 for s in submissions if s['annotation'] == 'authentic')
        forged = len(submissions) - authentic
        
        print(f"\n✅ Submission saved to: {OUTPUT_PATH}")
        print(f"   Total predictions: {len(submissions)}")
        print(f"   Authentic: {authentic}")
        print(f"   Forged: {forged}")
        
        # Show sample predictions
        print(f"\n📋 Sample predictions (first 5):")
        for i, row in enumerate(submissions[:5]):
            annotation = row['annotation']
            if len(annotation) > 50:
                annotation = annotation[:50] + "..."
            print(f"   {row['case_id']}: {annotation}")
        
        # Verify submission format
        print(f"\n🔍 Submission format check:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        print(f"   File exists: {OUTPUT_PATH.exists() if hasattr(OUTPUT_PATH, 'exists') else Path(OUTPUT_PATH).exists()}")
    else:
        print("❌ No test images found! Cannot generate submission.")
    
    print("\n"+"="*60)
    print("✅ COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()