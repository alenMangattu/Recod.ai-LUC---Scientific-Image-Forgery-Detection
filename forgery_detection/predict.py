"""
Prediction with Test-Time Augmentation (TTA)
=============================================
Generate predictions for test images with TTA for better accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd


class TTAPredictor:
    """
    Test-Time Augmentation predictor.
    Averages predictions over multiple augmented versions.
    """
    
    def __init__(self, model, device, tta_transforms=8):
        """
        Args:
            model: Trained model
            device: torch device
            tta_transforms: Number of TTA transforms (1, 2, 4, or 8)
                1 = no TTA
                2 = original + horizontal flip
                4 = original + flips
                8 = original + flips + 90° rotations
        """
        self.model = model
        self.device = device
        self.tta_transforms = tta_transforms
        self.model.eval()
    
    def predict_single(self, image_tensor):
        """
        Predict with TTA for a single image.
        
        Args:
            image_tensor: (1, 3, H, W) tensor
        
        Returns:
            (1, 1, H, W) prediction tensor
        """
        predictions = []
        
        # Original
        with torch.no_grad():
            pred = self.model(image_tensor)
            predictions.append(pred)
        
        if self.tta_transforms >= 2:
            # Horizontal flip
            img_flip_h = torch.flip(image_tensor, [3])
            with torch.no_grad():
                pred = self.model(img_flip_h)
                pred = torch.flip(pred, [3])  # Flip back
                predictions.append(pred)
        
        if self.tta_transforms >= 4:
            # Vertical flip
            img_flip_v = torch.flip(image_tensor, [2])
            with torch.no_grad():
                pred = self.model(img_flip_v)
                pred = torch.flip(pred, [2])
                predictions.append(pred)
            
            # Both flips
            img_flip_both = torch.flip(image_tensor, [2, 3])
            with torch.no_grad():
                pred = self.model(img_flip_both)
                pred = torch.flip(pred, [2, 3])
                predictions.append(pred)
        
        if self.tta_transforms >= 8:
            # 90° rotation
            img_rot90 = torch.rot90(image_tensor, 1, [2, 3])
            with torch.no_grad():
                pred = self.model(img_rot90)
                pred = torch.rot90(pred, -1, [2, 3])
                predictions.append(pred)
            
            # 90° + horizontal flip
            img_rot90_flip = torch.flip(img_rot90, [3])
            with torch.no_grad():
                pred = self.model(img_rot90_flip)
                pred = torch.flip(pred, [3])
                pred = torch.rot90(pred, -1, [2, 3])
                predictions.append(pred)
            
            # 270° rotation
            img_rot270 = torch.rot90(image_tensor, -1, [2, 3])
            with torch.no_grad():
                pred = self.model(img_rot270)
                pred = torch.rot90(pred, 1, [2, 3])
                predictions.append(pred)
            
            # 270° + horizontal flip
            img_rot270_flip = torch.flip(img_rot270, [3])
            with torch.no_grad():
                pred = self.model(img_rot270_flip)
                pred = torch.flip(pred, [3])
                pred = torch.rot90(pred, 1, [2, 3])
                predictions.append(pred)
        
        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred


def rle_encode(mask):
    """
    Run-length encode a binary mask (official competition format).
    
    Args:
        mask: Binary mask (numpy array H x W) where 1 = forgery
    
    Returns:
        RLE string or "authentic" if no forgery
    """
    # Transpose and flatten (matches official implementation)
    pixels = mask.T.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    if pixels.sum() == 0:
        return "authentic"
    
    # Find foreground pixels
    dots = np.where(pixels == 1)[0]
    
    # Build RLE pairs
    run_lengths = []
    prev = -2
    for b in dots:
        b = int(b)
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])  # 1-indexed
        run_lengths[-1] = int(run_lengths[-1] + 1)
        prev = b
    
    # Format as JSON string
    run_lengths = [int(x) for x in run_lengths]
    return json.dumps(run_lengths)


def generate_submission(model, test_dir, output_file, device,
                        target_size=512, threshold=0.5, tta=8):
    """
    Generate submission CSV for Kaggle.
    
    Args:
        model: Trained model
        test_dir: Directory with test images
        output_file: Output CSV path
        device: torch device
        target_size: Image size for model
        threshold: Binarization threshold
        tta: Number of TTA transforms (1, 2, 4, 8)
    
    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    print(f"TTA: {tta}x | Threshold: {threshold} | Size: {target_size}")
    
    # Setup predictor
    predictor = TTAPredictor(model, device, tta_transforms=tta)
    
    # Get test images
    test_dir = Path(test_dir)
    test_images = sorted(list(test_dir.glob('*.png')))
    print(f"Found {len(test_images)} test images")
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    predictions = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        case_id = img_path.stem
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (W, H)
            
            # Preprocess
            image = image.resize((target_size, target_size), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image = image.unsqueeze(0).to(device)
            
            # Predict with TTA
            pred = predictor.predict_single(image)
            
            # Resize to original size
            pred = F.interpolate(
                pred, size=(original_size[1], original_size[0]),
                mode='bilinear', align_corners=False
            )
            
            # Binarize
            mask = pred[0, 0].cpu().numpy()
            mask_binary = (mask > threshold).astype(np.uint8)
            
            # Encode
            annotation = rle_encode(mask_binary)
            
            predictions.append({
                'case_id': case_id,
                'annotation': annotation
            })
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            predictions.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    
    # Statistics
    authentic_count = sum(1 for p in predictions if p['annotation'] == 'authentic')
    forged_count = len(predictions) - authentic_count
    
    print(f"\n✅ Submission saved to: {output_file}")
    print(f"   Total: {len(predictions)}")
    print(f"   Authentic: {authentic_count}")
    print(f"   Forged: {forged_count}")
    
    return df


def predict_single_image(model, image_path, device, target_size=512, 
                         threshold=0.5, tta=8):
    """
    Predict for a single image.
    
    Returns:
        mask: Binary mask (H, W) numpy array
        prob_map: Probability map (H, W) numpy array
    """
    predictor = TTAPredictor(model, device, tta_transforms=tta)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image = image.resize((target_size, target_size), Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - mean) / std
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image = image.unsqueeze(0).to(device)
    
    # Predict
    pred = predictor.predict_single(image)
    
    # Resize to original
    pred = F.interpolate(
        pred, size=(original_size[1], original_size[0]),
        mode='bilinear', align_corners=False
    )
    
    prob_map = pred[0, 0].cpu().numpy()
    mask = (prob_map > threshold).astype(np.uint8)
    
    return mask, prob_map


if __name__ == '__main__':
    print("Prediction module loaded!")
    print("Use generate_submission() to create submission.csv")
