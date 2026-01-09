"""
Kaggle Competition Script - Forgery Detection
==============================================
Complete pipeline for Kaggle. Copy this to a Kaggle notebook cell.

Prerequisites:
1. Upload pretrained weights to Kaggle as dataset 'pretrained-weights'
2. Add competition data: 'recodai-luc-scientific-image-forgery-detection'

Usage in Kaggle notebook:
    %run kaggle_train.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models import resnet101
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths (Kaggle)
    'train_masks_dir': '/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_masks',
    'train_images_dir': '/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_images',
    'test_images_dir': '/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images',
    'weights_path': '/kaggle/input/pretrained-weights/resnet101_imagenet.pth',
    'output_dir': '/kaggle/working',
    
    # Model
    'target_size': 512,  # Higher = better but slower
    
    # Training
    'epochs': 50,
    'batch_size': 8,  # P100 16GB can handle 8 at 512x512
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'val_split': 0.15,
    'patience': 10,
    'num_workers': 4,
    
    # Prediction
    'tta': 8,  # Test-time augmentation (1, 2, 4, 8)
    'threshold': 0.5,
}

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# MODEL
# ============================================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.attention = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_ch // 4, skip_ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = skip * self.attention(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class FinalDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
    
    def forward(self, x):
        return self.conv2(self.conv1(self.upsample(x)))


class ResNet101UNet(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        
        backbone = resnet101(weights=None)
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained weights: {weights_path}")
            backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print("✓ Pretrained weights loaded!")
        else:
            print("⚠ No pretrained weights - training from scratch")
        
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4
        
        self.decoder5 = DecoderBlock(2048, 1024, 512)
        self.decoder4 = DecoderBlock(512, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = FinalDecoder(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        self._init_decoder()
    
    def _init_decoder(self):
        for m in [self.decoder5, self.decoder4, self.decoder3, self.decoder2, 
                  self.decoder1, self.final_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        e1 = self.encoder1(x)
        e1_pooled = self.pool(e1)
        e2 = self.encoder2(e1_pooled)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        d5 = self.decoder5(e5, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


# ============================================================================
# LOSSES
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        # Dice
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        
        # Focal
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = 0.25 * (1 - pt) ** 2 * bce
        
        return dice_loss + focal_loss.mean() + 0.5 * self.bce(pred, target)


# ============================================================================
# DATASET
# ============================================================================

class ForgeryDataset(Dataset):
    def __init__(self, mask_files, images_dir, target_size=512, augment=True):
        self.target_size = target_size
        self.augment = augment
        
        images_dir = Path(images_dir)
        forged_dir = images_dir / 'forged'
        authentic_dir = images_dir / 'authentic'
        
        self.valid_pairs = []
        for mf in mask_files:
            mid = Path(mf).stem
            fp = forged_dir / f"{mid}.png"
            ap = authentic_dir / f"{mid}.png"
            if fp.exists():
                self.valid_pairs.append((mf, fp))
            elif ap.exists():
                self.valid_pairs.append((mf, ap))
        
        print(f"Dataset: {len(self.valid_pairs)} pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        mask_file, image_path = self.valid_pairs[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = np.load(mask_file)
        if len(mask.shape) == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:,:,0]
        if mask.max() > 1:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.3:
                angle = random.choice([90, 180, 270])
                image = image.rotate(angle)
                mask = mask.rotate(angle)
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        
        image = np.array(image).astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        mask = np.array(mask).astype(np.float32) / 255.0
        
        return (torch.from_numpy(image.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask).unsqueeze(0).float())


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config):
    print("=" * 60)
    print("TRAINING FORGERY DETECTION MODEL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Data
    mask_files = sorted(list(Path(config['train_masks_dir']).glob('*.npy')))
    random.shuffle(mask_files)
    val_size = int(len(mask_files) * config['val_split'])
    train_files, val_files = mask_files[val_size:], mask_files[:val_size]
    
    train_dataset = ForgeryDataset(train_files, config['train_images_dir'], 
                                   config['target_size'], augment=True)
    val_dataset = ForgeryDataset(val_files, config['train_images_dir'],
                                 config['target_size'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'], 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = ResNet101UNet(weights_path=config['weights_path']).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['epochs']//3, 
                                            T_mult=1, eta_min=config['learning_rate']/100)
    
    best_f1, best_state = 0, None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss, intersection, pred_sum, gt_sum = 0, 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                pred = (outputs > 0.5).float()
                intersection += (pred * masks).sum().item()
                pred_sum += pred.sum().item()
                gt_sum += masks.sum().item()
        
        val_loss /= len(val_loader)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) * 100
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1)
        
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {f1:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ✓ New best F1: {best_f1:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


# ============================================================================
# PREDICTION WITH TTA
# ============================================================================

def predict_with_tta(model, image_tensor, device, tta=8):
    """Predict with test-time augmentation."""
    predictions = []
    
    with torch.no_grad():
        # Original
        predictions.append(model(image_tensor))
        
        if tta >= 2:
            # H flip
            pred = model(torch.flip(image_tensor, [3]))
            predictions.append(torch.flip(pred, [3]))
        
        if tta >= 4:
            # V flip
            pred = model(torch.flip(image_tensor, [2]))
            predictions.append(torch.flip(pred, [2]))
            # Both
            pred = model(torch.flip(image_tensor, [2, 3]))
            predictions.append(torch.flip(pred, [2, 3]))
        
        if tta >= 8:
            # Rotations
            for k in [1, -1]:
                img_rot = torch.rot90(image_tensor, k, [2, 3])
                pred = model(img_rot)
                predictions.append(torch.rot90(pred, -k, [2, 3]))
                
                img_rot_flip = torch.flip(img_rot, [3])
                pred = model(img_rot_flip)
                pred = torch.flip(pred, [3])
                predictions.append(torch.rot90(pred, -k, [2, 3]))
    
    return torch.stack(predictions).mean(dim=0)


def rle_encode(mask):
    """RLE encode mask."""
    pixels = mask.T.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    if pixels.sum() == 0:
        return "authentic"
    
    dots = np.where(pixels == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        b = int(b)
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])
        run_lengths[-1] += 1
        prev = b
    
    return json.dumps([int(x) for x in run_lengths])


def generate_submission(model, config):
    """Generate submission.csv"""
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    test_dir = Path(config['test_images_dir'])
    test_images = sorted(list(test_dir.glob('*.png')))
    print(f"Test images: {len(test_images)}")
    print(f"TTA: {config['tta']}x | Threshold: {config['threshold']}")
    
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    predictions = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        try:
            image = Image.open(img_path).convert('RGB')
            orig_size = image.size
            
            image = image.resize((config['target_size'], config['target_size']), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            
            pred = predict_with_tta(model, image, device, config['tta'])
            pred = F.interpolate(pred, size=(orig_size[1], orig_size[0]), 
                                mode='bilinear', align_corners=False)
            
            mask = pred[0, 0].cpu().numpy()
            mask_binary = (mask > config['threshold']).astype(np.uint8)
            
            predictions.append({
                'case_id': img_path.stem,
                'annotation': rle_encode(mask_binary)
            })
        except Exception as e:
            print(f"Error {img_path.stem}: {e}")
            predictions.append({'case_id': img_path.stem, 'annotation': 'authentic'})
    
    df = pd.DataFrame(predictions)
    output_path = Path(config['output_dir']) / 'submission.csv'
    df.to_csv(output_path, index=False)
    
    authentic = sum(1 for p in predictions if p['annotation'] == 'authentic')
    print(f"\n✓ Saved: {output_path}")
    print(f"  Authentic: {authentic}, Forged: {len(predictions) - authentic}")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n🚀 FORGERY DETECTION - COMPETITION PIPELINE")
    print("=" * 60)
    
    # Check paths
    if not os.path.exists(CONFIG['weights_path']):
        print(f"⚠ Weights not found: {CONFIG['weights_path']}")
        print("  Upload pretrained weights as Kaggle dataset!")
        print("  Or run download_weights.py locally first.")
    
    # Train
    model, history = train_model(CONFIG)
    
    # Save model
    torch.save(model.state_dict(), Path(CONFIG['output_dir']) / 'model.pth')
    print(f"✓ Model saved")
    
    # Generate submission
    submission = generate_submission(model, CONFIG)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"Submit: {CONFIG['output_dir']}/submission.csv")
    
    return model, history, submission


if __name__ == '__main__':
    model, history, submission = main()
