#!/usr/bin/env python3
"""
Prediction script for Kaggle Recod.ai/LUC - Scientific Image Forgery Detection
Generates submission.csv from test images using trained model.

Usage:
    # Activate virtual environment first
    source venv/bin/activate  # or: .venv/bin/activate
    
    # Basic usage (uses defaults)
    python predict.py
    
    # With custom paths
    python predict.py --model forgery_detection_model.pth --test_dir test_images --output submission.csv
    
    # Try different thresholds to optimize F1 score
    python predict.py --threshold 0.3  # Lower = more sensitive (higher recall)
    python predict.py --threshold 0.7  # Higher = more conservative (higher precision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class UNetEncoder(nn.Module):
    """Encoder using pre-trained ResNet."""
    def __init__(self, pretrained=True):
        super().__init__()
        # Use pre-trained ResNet34 as encoder
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
    def forward(self, x):
        # Store skip connections
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip1 = x  # 64 channels
        x = self.maxpool(x)
        
        x = self.layer1(x)
        skip2 = x  # 64 channels
        x = self.layer2(x)
        skip3 = x  # 128 channels
        x = self.layer3(x)
        skip4 = x  # 256 channels
        x = self.layer4(x)  # 512 channels
        
        return x, skip1, skip2, skip3, skip4

class UNetDecoder(nn.Module):
    """Decoder for U-Net."""
    def __init__(self):
        super().__init__()
        # Upsampling layers - need to handle size mismatches with interpolation
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x, skip1, skip2, skip3, skip4):
        # Decoder with skip connections
        # Up1: 16x16 -> 32x32, concat with skip4 (32x32)
        x = self.up1(x)
        if x.shape[2:] != skip4.shape[2:]:
            x = nn.functional.interpolate(x, size=skip4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        x = self.conv1(x)
        
        # Up2: 32x32 -> 64x64, concat with skip3 (64x64)
        x = self.up2(x)
        if x.shape[2:] != skip3.shape[2:]:
            x = nn.functional.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv2(x)
        
        # Up3: 64x64 -> 128x128, concat with skip2 (128x128)
        x = self.up3(x)
        if x.shape[2:] != skip2.shape[2:]:
            x = nn.functional.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)
        
        # Up4: 128x128 -> 256x256, concat with skip1 (256x256)
        x = self.up4(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = nn.functional.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv4(x)
        
        x = self.final(x)
        return x

class UNetSegmentation(nn.Module):
    """Complete U-Net with pre-trained ResNet encoder."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder()
        
    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, skip4)
        return torch.sigmoid(x)  # Output probabilities

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image_rgb(image_path):
    """Load image and convert to RGB format."""
    img_pil = Image.open(image_path)
    if img_pil.mode == 'RGBA':
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[3])
        return np.array(background)
    return np.array(img_pil.convert('RGB'))

# ============================================================================
# RLE ENCODING
# ============================================================================

def rle_encode(mask):
    """
    Run-length encode a binary mask.
    
    Args:
        mask: Binary mask (numpy array) where 1 = forgery, 0 = authentic
    
    Returns:
        RLE string like "[123 4 56 7]" or "authentic" if no forgery
    """
    # Flatten mask and convert to binary
    pixels = mask.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    # If no forgery pixels, return "authentic"
    if pixels.sum() == 0:
        return "authentic"
    
    # Find where pixels change
    diff = np.diff(pixels, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Create RLE pairs: [start, length, start, length, ...]
    rle_pairs = []
    for start, end in zip(starts, ends):
        length = end - start
        rle_pairs.extend([int(start), int(length)])
    
    # Format as string: "[123 4 56 7]"
    rle_str = "[" + " ".join(map(str, rle_pairs)) + "]"
    return rle_str

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_image(model, image_path, device, target_size=256, threshold=0.5):
    """
    Predict forgery mask for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        device: torch device
        target_size: Size to resize image for model
        threshold: Threshold for binarizing predictions
    
    Returns:
        Binary mask (numpy array) and original image size
    """
    # Load and preprocess image
    image = load_image_rgb(image_path)
    original_size = image.shape[:2]  # (H, W)
    
    # Resize for model
    image_pil = Image.fromarray(image)
    image_resized = image_pil.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to tensor
    image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)  # (1, 1, H, W)
        
        # Resize back to original size
        if output.shape[2:] != original_size:
            output = F.interpolate(
                output, size=original_size, 
                mode='bilinear', align_corners=False
            )
        
        # Convert to numpy and binarize
        mask = output[0, 0].cpu().numpy()  # (H, W)
        mask_binary = (mask > threshold).astype(np.uint8)
    
    return mask_binary, original_size

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_submission(model_path='forgery_detection_model.pth', 
                       test_dir='test_images',
                       output_file='submission.csv',
                       threshold=0.5):
    """
    Generate submission.csv from test images.
    
    Args:
        model_path: Path to trained model checkpoint
        test_dir: Directory containing test images
        output_file: Output CSV file path
        threshold: Threshold for binarizing predictions (0.5 default, try 0.3-0.7)
    """
    print("="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using CUDA (NVIDIA GPU): {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model architecture
    model = UNetSegmentation(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Get test images
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return None
    
    test_images = sorted(test_dir.glob('*.png'))
    print(f"\nFound {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return None
    
    # Generate predictions
    print(f"\nGenerating predictions (threshold={threshold})...")
    print("-" * 60)
    
    predictions = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        case_id = img_path.stem  # Get filename without extension
        
        try:
            # Predict
            mask_binary, original_size = predict_image(
                model, img_path, device, threshold=threshold
            )
            
            # Encode to RLE
            annotation = rle_encode(mask_binary)
            
            predictions.append({
                'case_id': case_id,
                'annotation': annotation
            })
            
        except Exception as e:
            print(f"❌ Error processing {case_id}: {e}")
            # Default to authentic on error
            predictions.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✅ Submission saved to: {output_file}")
    print(f"   Total predictions: {len(predictions)}")
    
    # Statistics
    authentic_count = sum(1 for p in predictions if p['annotation'] == 'authentic')
    forged_count = len(predictions) - authentic_count
    print(f"   Authentic: {authentic_count}")
    print(f"   Forged: {forged_count}")
    
    # Show first few predictions
    print("\nFirst 5 predictions:")
    print(df.head().to_string())
    
    return df

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Kaggle submission for forgery detection')
    parser.add_argument('--model', type=str, default='forgery_detection_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', type=str, default='test_images',
                       help='Directory containing test images')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binarizing predictions (0.3-0.7)')
    
    args = parser.parse_args()
    
    print("🚀 Starting prediction generation...")
    submission_df = generate_submission(
        model_path=args.model,
        test_dir=args.test_dir,
        output_file=args.output,
        threshold=args.threshold
    )
    
    if submission_df is not None:
        print("\n✅ Done! Upload submission.csv to Kaggle")
    else:
        print("\n❌ Failed to generate submission")

