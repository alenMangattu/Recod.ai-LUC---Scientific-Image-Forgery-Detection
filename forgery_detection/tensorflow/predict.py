"""
Prediction and Submission Generation - TensorFlow/Keras
========================================================
Includes Test-Time Augmentation (TTA) for improved predictions.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

from dataset import IMAGENET_MEAN, IMAGENET_STD, TestDataset


def predict_single(model, image, target_size=384):
    """
    Predict mask for a single image.
    
    Args:
        model: Trained model
        image: PIL Image or numpy array
        target_size: Model input size
    
    Returns:
        Predicted mask as numpy array
    """
    if isinstance(image, Image.Image):
        image = image.resize((target_size, target_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Predict
    pred = model(image, training=False)
    
    return pred[0].numpy()


def predict_tta(model, image_np, tta_level=8):
    """
    Test-Time Augmentation prediction.
    
    Args:
        model: Trained model
        image_np: Preprocessed numpy array (H, W, C)
        tta_level: Number of augmentations (1, 2, 4, or 8)
    
    Returns:
        Averaged prediction mask
    """
    preds = []
    
    # Original
    img = np.expand_dims(image_np, axis=0)
    pred = model(img, training=False)
    preds.append(pred[0].numpy())
    
    if tta_level >= 2:
        # Horizontal flip
        img_flipped = np.expand_dims(np.fliplr(image_np), axis=0)
        pred = model(img_flipped, training=False)
        preds.append(np.fliplr(pred[0].numpy()))
    
    if tta_level >= 4:
        # Vertical flip
        img_flipped = np.expand_dims(np.flipud(image_np), axis=0)
        pred = model(img_flipped, training=False)
        preds.append(np.flipud(pred[0].numpy()))
        
        # Both flips
        img_flipped = np.expand_dims(np.flipud(np.fliplr(image_np)), axis=0)
        pred = model(img_flipped, training=False)
        preds.append(np.flipud(np.fliplr(pred[0].numpy())))
    
    if tta_level >= 8:
        # 90 degree rotation
        img_rot = np.expand_dims(np.rot90(image_np, k=1), axis=0)
        pred = model(img_rot, training=False)
        preds.append(np.rot90(pred[0].numpy(), k=-1))
        
        # 270 degree rotation
        img_rot = np.expand_dims(np.rot90(image_np, k=3), axis=0)
        pred = model(img_rot, training=False)
        preds.append(np.rot90(pred[0].numpy(), k=-3))
    
    # Average predictions
    return np.mean(preds, axis=0)


def rle_encode(mask):
    """
    Run-length encode a binary mask.
    
    Args:
        mask: Binary mask array (H, W) or (H, W, 1)
    
    Returns:
        RLE string or 'authentic' if no forgery detected
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Flatten in column-major order (Fortran order)
    pixels = mask.T.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    if pixels.sum() == 0:
        return "authentic"
    
    # Find runs
    runs = []
    prev = -2
    for i in np.where(pixels == 1)[0]:
        i = int(i)
        if i > prev + 1:
            runs.extend([i + 1, 0])
        runs[-1] += 1
        prev = i
    
    return json.dumps([int(x) for x in runs])


def generate_submission(model, config, test_images_dir, output_dir):
    """
    Generate submission CSV file.
    
    Args:
        model: Trained model
        config: Configuration dict with target_size, tta, threshold
        test_images_dir: Directory with test images
        output_dir: Output directory for submission.csv
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    
    return generate_submission_from_sample(
        model=model,
        config=config,
        test_images_dir=test_images_dir,
        output_dir=output_dir,
        sample_submission_path=config.get(
            "sample_submission_path",
            "/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv",
        ),
        fail_on_error=config.get("fail_on_error", True),
    )


def _index_test_images(test_dir: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> dict[str, Path]:
    # One pass scan. Handles nested layouts and multiple extensions.
    idx: dict[str, Path] = {}
    for ext in exts:
        for p in test_dir.rglob(f"*{ext}"):
            idx[p.stem] = p
    return idx


def generate_submission_from_sample(
    model,
    config,
    test_images_dir,
    output_dir,
    sample_submission_path=None,
    fail_on_error: bool = True,
):
    """
    Generate submission CSV driven by sample_submission.csv (authoritative case_id list).
    """
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images_dir not found: {test_images_dir}")

    image_index = _index_test_images(test_dir)

    case_ids = None
    if sample_submission_path:
        sample_path = Path(sample_submission_path)
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            if "case_id" not in sample_df.columns:
                raise ValueError(f"sample_submission missing 'case_id' column: {sample_path}")
            case_ids = [str(x) for x in sample_df["case_id"].tolist()]

    if case_ids is None:
        # Fallback: whatever is present in test_dir (still recursive)
        case_ids = sorted(image_index.keys())

    print(f"Found {len(case_ids)} cases (sample-driven={sample_submission_path is not None})")
    print(f"TTA: {config['tta']}x, Threshold: {config['threshold']}")

    results = []
    errors = []

    for case_id in tqdm(case_ids, desc="Predicting"):
        img_path = image_index.get(str(case_id))
        if img_path is None:
            errors.append(f"Missing image for case_id={case_id}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  # (W, H)

            img_resized = image.resize((config["target_size"], config["target_size"]), Image.BILINEAR)
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

            pred = predict_tta(model, img_np, tta_level=config["tta"])
            pred_resized = tf.image.resize(pred, [original_size[1], original_size[0]], method="bilinear").numpy()

            mask = (pred_resized > config["threshold"]).astype(np.uint8)
            rle = rle_encode(mask)
            results.append({"case_id": str(case_id), "annotation": rle})

        except Exception as e:
            errors.append(f"case_id={case_id} error: {e}")

    if errors:
        msg = "Submission generation had errors:\n" + "\n".join(errors[:20])
        if len(errors) > 20:
            msg += f"\n... and {len(errors) - 20} more"
        if fail_on_error:
            raise RuntimeError(msg)
        else:
            print("[WARN] " + msg)
            # Fill missing cases as authentic only when explicitly allowed
            predicted_ids = {r["case_id"] for r in results}
            for cid in case_ids:
                if str(cid) not in predicted_ids:
                    results.append({"case_id": str(cid), "annotation": "authentic"})
    
    # Save submission
    df = pd.DataFrame(results)
    output_path = Path(output_dir) / 'submission.csv'
    df.to_csv(output_path, index=False)
    
    # Statistics
    authentic_count = sum(1 for r in results if r['annotation'] == 'authentic')
    forged_count = len(results) - authentic_count
    
    print(f"\nSaved: {output_path}")
    print(f"Authentic: {authentic_count}, Forged: {forged_count}")
    
    return df


def load_model_for_inference(model_path, config):
    """
    Load a saved model for inference.
    
    Args:
        model_path: Path to saved model (.keras or .h5)
        config: Configuration dict
    
    Returns:
        Loaded model
    """
    from model import build_model
    
    model_path = Path(model_path)
    
    if model_path.suffix in ['.keras', '.h5']:
        # Load full model
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        # Load weights only
        model = build_model(
            input_shape=(config['target_size'], config['target_size'], 3)
        )
        model.load_weights(str(model_path))
    
    return model


def main():
    """Main entry point for prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Submission')
    parser.add_argument('--model_path', default='/kaggle/working/model.keras')
    parser.add_argument('--test_images_dir', default='/kaggle/input/forgery-detection/test_images')
    parser.add_argument('--output_dir', default='/kaggle/working')
    parser.add_argument('--target_size', type=int, default=384)
    parser.add_argument('--tta', type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()
    
    config = {
        'target_size': args.target_size,
        'tta': args.tta,
        'threshold': args.threshold,
    }
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model_for_inference(args.model_path, config)
    
    # Generate submission
    df = generate_submission(model, config, args.test_images_dir, args.output_dir)
    
    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
