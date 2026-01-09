#!/usr/bin/env python3
"""
Local Test Script for Forgery Detection
========================================
Tests the model on your local data before uploading to Kaggle.

Usage:
    python run_local.py --train_masks_dir ../train_masks --train_images_dir ../train_images

Or with defaults (assumes data in parent directory):
    python run_local.py
"""

import os
import sys
import argparse
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
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config(args):
    """Get configuration based on command line args."""
    return {
        # Paths
        'train_masks_dir': args.train_masks_dir,
        'train_images_dir': args.train_images_dir,
        'test_images_dir': args.test_images_dir,
        'weights_path': args.weights_path,
        'output_dir': args.output_dir,
        
        # Model
        'target_size': args.target_size,
        
        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'val_split': 0.15,
        'patience': args.patience,
        'num_workers': args.num_workers,
        
        # Prediction
        'tta': args.tta,
        'threshold': args.threshold,
    }

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            nn.Conv2d(skip_ch, max(1, skip_ch // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, skip_ch // 4), skip_ch, 1),
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
            print(f"✓ Loading pretrained weights: {weights_path}")
            backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
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
        
        print(f"  Dataset: {len(self.valid_pairs)} valid pairs")
    
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
    print("\n" + "=" * 60)
    print("TRAINING FORGERY DETECTION MODEL")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (slow!)")
    
    # Data
    print(f"\nLoading data from: {config['train_masks_dir']}")
    mask_files = sorted(list(Path(config['train_masks_dir']).glob('*.npy')))
    if len(mask_files) == 0:
        print(f"❌ No mask files found in {config['train_masks_dir']}")
        return None, None
    
    random.shuffle(mask_files)
    val_size = int(len(mask_files) * config['val_split'])
    train_files, val_files = mask_files[val_size:], mask_files[:val_size]
    
    print(f"  Found {len(mask_files)} mask files")
    print(f"  Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_dataset = ForgeryDataset(train_files, config['train_images_dir'], 
                                   config['target_size'], augment=True)
    val_dataset = ForgeryDataset(val_files, config['train_images_dir'],
                                 config['target_size'], augment=False)
    
    if len(train_dataset) == 0:
        print("❌ No valid image-mask pairs found!")
        return None, None
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'], 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True)
    
    # Model
    print(f"\nCreating model...")
    model = ResNet101UNet(weights_path=config['weights_path']).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, config['epochs']//3), 
                                            T_mult=1, eta_min=config['learning_rate']/100)
    
    best_f1, best_state = 0, None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    print(f"\nTraining for {config['epochs']} epochs...")
    print("-" * 60)
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
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
        
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {f1:.2f}%", end="")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(" ✓ Best!")
        else:
            patience_counter += 1
            print()
            if patience_counter >= config['patience']:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✓ Loaded best model (F1: {best_f1:.2f}%)")
    
    return model, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, config, num_samples=4):
    """Visualize some predictions."""
    print("\n" + "=" * 60)
    print("VISUALIZING PREDICTIONS")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get some validation samples
    mask_files = sorted(list(Path(config['train_masks_dir']).glob('*.npy')))[:num_samples]
    
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, mask_file in enumerate(mask_files):
        mask_id = mask_file.stem
        
        # Find image
        images_dir = Path(config['train_images_dir'])
        image_path = images_dir / 'forged' / f"{mask_id}.png"
        if not image_path.exists():
            image_path = images_dir / 'authentic' / f"{mask_id}.png"
        
        if not image_path.exists():
            continue
        
        # Load
        image = Image.open(image_path).convert('RGB')
        mask = np.load(mask_file)
        if len(mask.shape) == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:,:,0]
        
        # Preprocess
        img_resized = image.resize((config['target_size'], config['target_size']), Image.BILINEAR)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_norm = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(img_tensor)
        pred = pred[0, 0].cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f'Image: {mask_id}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = Path(config['output_dir']) / 'predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    plt.show()


def plot_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_f1'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score (%)')
    axes[1].set_title('Validation F1 Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training history to: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Forgery Detection Locally')
    
    # Paths
    parser.add_argument('--train_masks_dir', type=str, default='../train_masks',
                        help='Directory with training masks (.npy files)')
    parser.add_argument('--train_images_dir', type=str, default='../train_images',
                        help='Directory with training images')
    parser.add_argument('--test_images_dir', type=str, default='../test_images',
                        help='Directory with test images')
    parser.add_argument('--weights_path', type=str, 
                        default='pretrained_weights/resnet101_imagenet.pth',
                        help='Path to pretrained weights')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10 for quick test)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4 for local testing)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--target_size', type=int, default=256,
                        help='Image size (default: 256 for quick test, use 512 for full)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for debugging)')
    
    # Prediction
    parser.add_argument('--tta', type=int, default=1,
                        help='TTA transforms (1=none, 8=full)')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    # Options
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (3 epochs, small size)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.epochs = 3
        args.target_size = 128
        args.batch_size = 2
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = get_config(args)
    
    print("\n" + "=" * 60)
    print("🚀 FORGERY DETECTION - LOCAL TEST")
    print("=" * 60)
    print(f"Train masks: {config['train_masks_dir']}")
    print(f"Train images: {config['train_images_dir']}")
    print(f"Weights: {config['weights_path']}")
    print(f"Output: {config['output_dir']}")
    print(f"Size: {config['target_size']}x{config['target_size']}")
    print(f"Epochs: {config['epochs']}, Batch: {config['batch_size']}")
    
    set_seed(42)
    
    # Check paths
    if not Path(config['train_masks_dir']).exists():
        print(f"\n❌ Train masks directory not found: {config['train_masks_dir']}")
        print("   Please specify correct path with --train_masks_dir")
        return
    
    if not Path(config['train_images_dir']).exists():
        print(f"\n❌ Train images directory not found: {config['train_images_dir']}")
        print("   Please specify correct path with --train_images_dir")
        return
    
    # Train
    model, history = train_model(config)
    
    if model is None:
        print("\n❌ Training failed!")
        return
    
    # Save model
    model_path = Path(config['output_dir']) / 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Plot history
    if history:
        plot_history(history, config['output_dir'])
    
    # Visualize predictions
    visualize_predictions(model, config, num_samples=4)
    
    print("\n" + "=" * 60)
    print("✅ LOCAL TEST COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Upload pretrained_weights/ folder to Kaggle as dataset")
    print(f"2. Copy kaggle_train.py to Kaggle notebook")
    print(f"3. Run on Kaggle with full settings (512x512, 50 epochs)")


if __name__ == '__main__':
    main()
