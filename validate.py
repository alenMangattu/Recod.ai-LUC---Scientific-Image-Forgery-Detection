"""
Validate script - Load pretrained model and generate predictions on unseen holdout data
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from pathlib import Path
import pandas as pd

# ============================================================================
# CONFIGURATION (must match training script)
# ============================================================================
TRAIN_IMAGES_DIR = Path('train_images')
TRAIN_MASKS_DIR = Path('train_masks')
MODEL_PATH = 'best_model.keras'
IMG_SIZE = (256, 256)
TRAIN_SIZE = None  # Must match training script (None = all data)
VAL_SIZE = 200  # Must match training script
HOLDOUT_SIZE = 100  # Must match training script
SPLIT_SEED = 1337  # Must match training script

# Optional: reduce TF logging noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================================
# CUSTOM LOSS AND METRICS (must match training)
# ============================================================================

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """Focal Tversky Loss - handles class imbalance better"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.pow(1 - tversky, gamma)

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

# ============================================================================
# DATA LOADING
# ============================================================================

def find_image(mask_name):
    """Find corresponding image for a mask"""
    name = mask_name.stem
    for folder in ['forged', 'authentic']:
        path = TRAIN_IMAGES_DIR / folder / f'{name}.png'
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

def predict_holdout(model, mask_files, holdout_indices, img_size=(256, 256), threshold=0.5):
    """Generate predictions for holdout indices (unseen data)"""
    submissions = []
    scores = {"iou": [], "pixel_acc": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    
    print(f"\nGenerating predictions for {len(holdout_indices)} holdout images (unseen)...")
    print(f"Using threshold: {threshold}")
    
    for i, idx in enumerate(holdout_indices):
        try:
            img, true_mask = getdata(idx, mask_files)
            original_shape = img.shape[:2]
            
            # Get case_id from mask filename
            mask_path = mask_files[idx]
            case_id = mask_path.stem
            
            # Preprocess
            img_resized = tf.image.resize(img, img_size)
            img_tensor = tf.expand_dims(img_resized / 255.0, 0)
            
            # Predict
            pred = model.predict(img_tensor, verbose=0)
            pred_raw = pred[0, :, :, 0]
            
            # Debug: show prediction stats for first few
            if i < 5:
                print(f"\n  [{case_id}] Pred stats: min={pred_raw.min():.4f}, max={pred_raw.max():.4f}, mean={pred_raw.mean():.4f}")
                print(f"       True mask: sum={true_mask.sum():.0f}, max={true_mask.max():.2f} (is forged: {true_mask.sum() > 0})")
            
            pred_mask = (pred_raw > threshold).astype(np.uint8)
            
            # Resize back to original
            pred_mask_resized = tf.image.resize(
                pred_mask[..., None], 
                original_shape, 
                method='nearest'
            ).numpy()[:, :, 0].astype(np.uint8)
            
            # Resize true mask to 256x256 for scoring (same as pred before resize)
            true_mask_resized = tf.image.resize(
                true_mask[..., None],
                img_size,
                method='nearest'
            ).numpy()[:, :, 0]
            true_mask_binary = (true_mask_resized > 0.5).astype(np.uint8)
            pred_mask_for_score = (pred_raw > threshold).astype(np.uint8)
            
            # Pixel accuracy
            pixel_acc = (pred_mask_for_score == true_mask_binary).mean()
            scores["pixel_acc"].append(pixel_acc)
            
            # IoU
            intersection = np.logical_and(pred_mask_for_score, true_mask_binary).sum()
            union = np.logical_or(pred_mask_for_score, true_mask_binary).sum()
            iou = intersection / union if union > 0 else 1.0
            scores["iou"].append(iou)
            
            # Classification (forged vs authentic)
            true_forged = true_mask.sum() > 0
            pred_forged = pred_mask_resized.sum() > 0
            
            if true_forged and pred_forged:
                scores["tp"] += 1
            elif not true_forged and not pred_forged:
                scores["tn"] += 1
            elif not true_forged and pred_forged:
                scores["fp"] += 1
            else:  # true_forged and not pred_forged
                scores["fn"] += 1
            
            # Create submission entry
            pred_sum = pred_mask_resized.sum()
            if pred_sum == 0:
                submissions.append({"case_id": case_id, "annotation": "authentic", "_pred_sum": 0})
            else:
                rle = rle_encode(pred_mask_resized)
                submissions.append({"case_id": case_id, "annotation": rle, "_pred_sum": pred_sum})
            
            if i < 5:
                print(f"       After threshold: {pred_sum} pixels detected (IoU: {iou:.4f}, PixelAcc: {pixel_acc:.4f})")
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            mask_path = mask_files[idx]
            case_id = mask_path.stem
            submissions.append({"case_id": case_id, "annotation": "authentic", "_pred_sum": 0})
            scores["fn"] += 1
    
    return submissions, scores

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Load model and generate predictions on unseen holdout data"""
    
    # Setup GPU
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {gpus}")
    else:
        print("No GPUs detected, using CPU")
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = keras.models.load_model(
        MODEL_PATH, 
        custom_objects={
            'focal_tversky_loss': focal_tversky_loss,
            'F1ScoreMetric': F1ScoreMetric,
            'f1_score': F1ScoreMetric  # Handle both class name and metric name
        }
    )
    print("Model loaded successfully!")
    
    # Load mask files
    print("\nLoading mask files...")
    mask_files = list(TRAIN_MASKS_DIR.glob('*.npy'))
    print(f"Found {len(mask_files)} mask files")
    
    # Get holdout indices using same split logic as training
    if TRAIN_SIZE is None:
        total_n = len(mask_files)
    else:
        total_n = min(TRAIN_SIZE, len(mask_files))

    if total_n < (VAL_SIZE + HOLDOUT_SIZE + 1):
        raise ValueError(
            f"Not enough training samples for splits: total_n={total_n}, "
            f"VAL_SIZE={VAL_SIZE}, HOLDOUT_SIZE={HOLDOUT_SIZE}"
        )

    # Deterministic shuffle (same as training)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(total_n)

    holdout_idx = perm[:HOLDOUT_SIZE].tolist()
    val_idx = perm[HOLDOUT_SIZE:HOLDOUT_SIZE + VAL_SIZE].tolist()
    train_idx = perm[HOLDOUT_SIZE + VAL_SIZE:].tolist()

    holdout_indices = holdout_idx

    print(f"\nUsing same split as training (seed={SPLIT_SEED}):")
    print(f"  Train:   {len(train_idx)} images")
    print(f"  Val:     {len(val_idx)} images")
    print(f"  Holdout: {len(holdout_indices)} images (UNSEEN - for submission)")
    print(f"\nHoldout indices: {holdout_indices}")
    
    # Generate predictions
    submissions, scores = predict_holdout(model, mask_files, holdout_indices, img_size=IMG_SIZE)
    
    # Save submission (remove debug field)
    for s in submissions:
        s.pop('_pred_sum', None)
    df = pd.DataFrame(submissions)
    df.to_csv('submission.csv', index=False)
    print(f"\n✓ Submission file saved: submission.csv")
    print(f"  Total predictions: {len(submissions)}")
    print(f"  Authentic predictions: {sum(1 for s in submissions if s['annotation'] == 'authentic')}")
    print(f"  Forgery predictions: {sum(1 for s in submissions if s['annotation'] != 'authentic')}")
    
    # Print scores
    print("\n" + "=" * 50)
    print("VALIDATION SCORES (on holdout data)")
    print("=" * 50)
    
    # Pixel-level metrics
    mean_iou = np.mean(scores["iou"]) if scores["iou"] else 0
    mean_pixel_acc = np.mean(scores["pixel_acc"]) if scores["pixel_acc"] else 0
    print(f"\nPixel-Level Metrics:")
    print(f"  Mean IoU:           {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"  Mean Pixel Accuracy: {mean_pixel_acc:.4f} ({mean_pixel_acc*100:.2f}%)")
    
    # Classification metrics
    tp, fp, tn, fn = scores["tp"], scores["fp"], scores["tn"], scores["fn"]
    total = tp + fp + tn + fn
    classification_acc = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nClassification Metrics (Forged vs Authentic):")
    print(f"  True Positives:  {tp} (correctly detected forgeries)")
    print(f"  True Negatives:  {tn} (correctly identified authentic)")
    print(f"  False Positives: {fp} (authentic marked as forged)")
    print(f"  False Negatives: {fn} (forgeries missed)")
    print(f"\n  Classification Accuracy: {classification_acc:.4f} ({classification_acc*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print("\n" + "=" * 50)
    print("Complete! submission.csv created from unseen holdout data.")
    print("=" * 50)

if __name__ == "__main__":
    main()

