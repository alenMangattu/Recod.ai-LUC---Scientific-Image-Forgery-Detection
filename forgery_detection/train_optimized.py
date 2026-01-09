#!/usr/bin/env python3
"""
WINNING SOLUTION - Forgery Detection Competition
=================================================
Optimized for top leaderboard performance.

Key optimizations:
1. Pretrained ResNet101 encoder
2. Proper warmup + cosine annealing LR
3. Heavy augmentation for copy-move detection
4. Focal + Dice loss for class imbalance
5. TTA for predictions
6. Optimized for 30 min local / 4 hour Kaggle

Usage:
    python train_optimized.py --quick      # 30 min local test
    python train_optimized.py --full       # Full training
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
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet101, resnet50
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config(args):
    if args.quick:
        return {
            'train_masks_dir': args.train_masks_dir,
            'train_images_dir': args.train_images_dir,
            'test_images_dir': args.test_images_dir,
            'weights_path': args.weights_path,
            'output_dir': args.output_dir,
            'target_size': 256,
            'epochs': 15,
            'batch_size': 4,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'val_split': 0.1,
            'patience': 5,
            'num_workers': 0,
            'tta': 4,
            'threshold': 0.4,
            'use_amp': False,  # Mixed precision
        }
    else:
        return {
            'train_masks_dir': args.train_masks_dir,
            'train_images_dir': args.train_images_dir,
            'test_images_dir': args.test_images_dir,
            'weights_path': args.weights_path,
            'output_dir': args.output_dir,
            'target_size': 384,  # Good balance
            'epochs': 40,
            'batch_size': 6,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'val_split': 0.1,
            'patience': 8,
            'num_workers': 4,
            'tta': 8,
            'threshold': 0.4,
            'use_amp': True,
        }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ============================================================================
# MODEL - Optimized U-Net with ResNet Encoder
# ============================================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections - focuses on relevant features."""
    def __init__(self, gate_ch, skip_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, skip_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, skip_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_ch, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        if g.shape[2:] != skip.shape[2:]:
            g = F.interpolate(g, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.W_skip(skip)
        attention = self.psi(g + x)
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_attention=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attention = AttentionGate(in_ch, skip_ch) if use_attention else None
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.attention:
            skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class ForgeryDetector(nn.Module):
    """
    U-Net with pretrained ResNet101 encoder + Attention gates.
    Optimized for copy-move forgery detection.
    """
    def __init__(self, weights_path=None, encoder='resnet101'):
        super().__init__()
        
        # Load encoder
        if encoder == 'resnet101':
            backbone = resnet101(weights=None)
            if weights_path and os.path.exists(weights_path):
                print(f"✓ Loading pretrained: {Path(weights_path).name}")
                backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
        else:
            backbone = resnet50(weights=None)
            if weights_path and os.path.exists(weights_path):
                print(f"✓ Loading pretrained: {Path(weights_path).name}")
                backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        # Encoder stages
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64, /2
        self.pool = backbone.maxpool  # /4
        self.enc2 = backbone.layer1  # 256, /4
        self.enc3 = backbone.layer2  # 512, /8
        self.enc4 = backbone.layer3  # 1024, /16
        self.enc5 = backbone.layer4  # 2048, /32
        
        # Decoder with attention
        self.dec5 = DecoderBlock(2048, 1024, 512)
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        
        # Final upsampling and output
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            ConvBnRelu(64, 32),
            ConvBnRelu(32, 32),
            nn.Conv2d(32, 1, 1)
        )
        
        # Deep supervision heads
        self.ds4 = nn.Conv2d(512, 1, 1)
        self.ds3 = nn.Conv2d(256, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.dec5, self.dec4, self.dec3, self.dec2, self.final_conv, self.ds4, self.ds3]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, return_all=False):
        size = x.shape[2:]
        
        # Encoder
        e1 = self.enc1(x)      # /2
        e2 = self.enc2(self.pool(e1))  # /4
        e3 = self.enc3(e2)     # /8
        e4 = self.enc4(e3)     # /16
        e5 = self.enc5(e4)     # /32
        
        # Decoder
        d5 = self.dec5(e5, e4)  # /16
        d4 = self.dec4(d5, e3)  # /8
        d3 = self.dec3(d4, e2)  # /4
        d2 = self.dec2(d3, e1)  # /2
        
        # Final
        out = self.final_up(d2)  # /1
        out = self.final_conv(out)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        
        if return_all:
            ds4 = F.interpolate(self.ds4(d5), size=size, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d4), size=size, mode='bilinear', align_corners=False)
            return torch.sigmoid(out), torch.sigmoid(ds4), torch.sigmoid(ds3)
        
        return torch.sigmoid(out)


# ============================================================================
# LOSS - Optimized for class imbalance
# ============================================================================

class ForgeryLoss(nn.Module):
    """Combined loss optimized for forgery detection."""
    def __init__(self, dice_w=1.0, focal_w=1.0, bce_w=0.5):
        super().__init__()
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.bce_w = bce_w
    
    def dice_loss(self, pred, target):
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def focal_loss(self, pred, target, gamma=2.0, alpha=0.75):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        # Higher alpha for positive class (forgery) to focus on detecting them
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal = alpha_t * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def forward(self, pred, target, deep_outputs=None):
        loss = self.dice_w * self.dice_loss(pred, target)
        loss += self.focal_w * self.focal_loss(pred, target)
        loss += self.bce_w * F.binary_cross_entropy(pred, target)
        
        # Deep supervision
        if deep_outputs is not None:
            for dp in deep_outputs:
                loss += 0.3 * self.dice_loss(dp, target)
        
        return loss


# ============================================================================
# DATASET - Heavy augmentation for copy-move detection
# ============================================================================

class ForgeryDataset(Dataset):
    def __init__(self, mask_files, images_dir, target_size=384, augment=True, training=True):
        self.target_size = target_size
        self.augment = augment and training
        self.training = training
        
        images_dir = Path(images_dir)
        forged_dir = images_dir / 'forged'
        authentic_dir = images_dir / 'authentic'
        
        self.pairs = []
        for mf in mask_files:
            mid = Path(mf).stem
            fp = forged_dir / f"{mid}.png"
            ap = authentic_dir / f"{mid}.png"
            if fp.exists():
                self.pairs.append((mf, fp))
            elif ap.exists():
                self.pairs.append((mf, ap))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        mask_file, image_path = self.pairs[idx]
        
        # Load
        image = Image.open(image_path).convert('RGB')
        mask = np.load(mask_file)
        if len(mask.shape) == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:,:,0]
        if mask.max() > 1:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Resize
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # To tensor
        image = np.array(image).astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        mask = np.array(mask).astype(np.float32) / 255.0
        
        return (torch.from_numpy(image.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask).unsqueeze(0).float())
    
    def _augment(self, image, mask):
        # Flips (50%)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Rotation 90/180/270 (30%)
        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        
        # Random rotation -20 to 20 (30%)
        if random.random() < 0.3:
            angle = random.uniform(-20, 20)
            image = image.rotate(angle, Image.BILINEAR, fillcolor=(128, 128, 128))
            mask = mask.rotate(angle, Image.NEAREST, fillcolor=0)
        
        # Color jitter (50%)
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        
        # Blur (20%) - helps detect compression artifacts
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Random crop & resize (30%)
        if random.random() < 0.3:
            w, h = image.size
            crop_size = random.uniform(0.75, 0.95)
            new_w, new_h = int(w * crop_size), int(h * crop_size)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)
            mask = mask.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.NEAREST)
        
        return image, mask


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config):
    print("\n" + "=" * 60)
    print("🚀 FORGERY DETECTION - OPTIMIZED TRAINING")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ MPS (Apple Silicon)")
        config['use_amp'] = False  # MPS doesn't support AMP well
    else:
        device = torch.device('cpu')
        print("⚠ CPU only")
        config['use_amp'] = False
    
    # Data
    mask_files = sorted(list(Path(config['train_masks_dir']).glob('*.npy')))
    print(f"✓ Found {len(mask_files)} samples")
    
    random.shuffle(mask_files)
    val_size = int(len(mask_files) * config['val_split'])
    train_files, val_files = mask_files[val_size:], mask_files[:val_size]
    
    train_ds = ForgeryDataset(train_files, config['train_images_dir'], config['target_size'], augment=True)
    val_ds = ForgeryDataset(val_files, config['train_images_dir'], config['target_size'], augment=False)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)
    
    # Model
    model = ForgeryDetector(weights_path=config['weights_path']).to(device)
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Training setup
    criterion = ForgeryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Warmup + Cosine annealing
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if config['use_amp'] else None
    
    best_f1, best_state = 0, None
    patience_cnt = 0
    
    print(f"\n{'Epoch':<6} {'Train':<10} {'Val':<10} {'F1':<10} {'LR':<12}")
    print("-" * 50)
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if config['use_amp']:
                with autocast():
                    out, ds4, ds3 = model(images, return_all=True)
                    loss = criterion(out, masks, [ds4, ds3])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out, ds4, ds3 = model(images, return_all=True)
                loss = criterion(out, masks, [ds4, ds3])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss, intersection, pred_sum, gt_sum = 0, 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                out = model(images)
                val_loss += criterion(out, masks).item()
                pred = (out > config['threshold']).float()
                intersection += (pred * masks).sum().item()
                pred_sum += pred.sum().item()
                gt_sum += masks.sum().item()
        
        val_loss /= len(val_loader)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) * 100
        
        lr = optimizer.param_groups[0]['lr']
        marker = ""
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            marker = " ✓"
        else:
            patience_cnt += 1
        
        print(f"{epoch+1:<6} {train_loss:<10.4f} {val_loss:<10.4f} {f1:<9.2f}% {lr:<12.6f}{marker}")
        
        if patience_cnt >= config['patience']:
            print(f"\n⏹ Early stopping")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n✓ Best F1: {best_f1:.2f}%")
    return model, best_f1


# ============================================================================
# PREDICTION WITH TTA
# ============================================================================

def predict_tta(model, image, device, tta=8):
    """Test-time augmentation."""
    preds = []
    
    with torch.no_grad():
        preds.append(model(image))
        
        if tta >= 2:
            preds.append(torch.flip(model(torch.flip(image, [3])), [3]))
        
        if tta >= 4:
            preds.append(torch.flip(model(torch.flip(image, [2])), [2]))
            preds.append(torch.flip(model(torch.flip(image, [2, 3])), [2, 3]))
        
        if tta >= 8:
            for k in [1, -1]:
                rotated = torch.rot90(image, k, [2, 3])
                pred = model(rotated)
                preds.append(torch.rot90(pred, -k, [2, 3]))
    
    return torch.stack(preds).mean(0)


def rle_encode(mask):
    """RLE encode with proper format."""
    pixels = mask.T.flatten()
    pixels = (pixels > 0.5).astype(np.uint8)
    
    if pixels.sum() == 0:
        return "authentic"
    
    runs = []
    prev = -2
    for i in np.where(pixels == 1)[0]:
        i = int(i)
        if i > prev + 1:
            runs.extend([i + 1, 0])
        runs[-1] += 1
        prev = i
    
    return json.dumps([int(x) for x in runs])


def generate_submission(model, config):
    """Generate submission CSV."""
    print("\n" + "=" * 60)
    print("📝 GENERATING SUBMISSION")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    test_dir = Path(config['test_images_dir'])
    test_images = sorted(list(test_dir.glob('*.png')))
    print(f"✓ {len(test_images)} test images")
    print(f"  TTA: {config['tta']}x, Threshold: {config['threshold']}")
    
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    results = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        try:
            image = Image.open(img_path).convert('RGB')
            orig_size = image.size
            
            img = image.resize((config['target_size'], config['target_size']), Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            
            pred = predict_tta(model, img, device, config['tta'])
            pred = F.interpolate(pred, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
            
            mask = pred[0, 0].cpu().numpy()
            mask = (mask > config['threshold']).astype(np.uint8)
            
            results.append({'case_id': img_path.stem, 'annotation': rle_encode(mask)})
        except Exception as e:
            print(f"Error {img_path.stem}: {e}")
            results.append({'case_id': img_path.stem, 'annotation': 'authentic'})
    
    df = pd.DataFrame(results)
    out_path = Path(config['output_dir']) / 'submission.csv'
    df.to_csv(out_path, index=False)
    
    auth = sum(1 for r in results if r['annotation'] == 'authentic')
    print(f"\n✓ Saved: {out_path}")
    print(f"  Authentic: {auth}, Forged: {len(results) - auth}")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_masks_dir', default='../train_masks')
    parser.add_argument('--train_images_dir', default='../train_images')
    parser.add_argument('--test_images_dir', default='../test_images')
    parser.add_argument('--weights_path', default='pretrained_weights/resnet101_imagenet.pth')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--quick', action='store_true', help='Quick 30min test')
    parser.add_argument('--full', action='store_true', help='Full training')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    
    config = get_config(args)
    
    print(f"\n📊 Config: {config['target_size']}px, {config['epochs']} epochs, batch={config['batch_size']}")
    
    # Train
    model, best_f1 = train_model(config)
    
    # Save
    torch.save(model.state_dict(), Path(config['output_dir']) / 'model.pth')
    
    # Generate submission if test images exist
    if Path(config['test_images_dir']).exists():
        generate_submission(model, config)
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE! Best F1: {best_f1:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
