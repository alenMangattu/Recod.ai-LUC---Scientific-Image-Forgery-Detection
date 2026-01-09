"""
Exact replica of the working Kaggle model architecture.
Train this properly and you'll get similar results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math


class DinoTinyDecoder(nn.Module):
    """Simple 3-layer decoder - exactly as in working code."""
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, f, size):
        return self.net(F.interpolate(f, size=size, mode="bilinear", align_corners=False))


class DinoSegmenter(nn.Module):
    """DINOv2 + simple decoder for forgery segmentation."""
    
    def __init__(self, model_name='dinov2_vitb14', img_size=512):
        super().__init__()
        self.img_size = img_size
        
        # Load DINOv2 from torch hub
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        self.patch_size = 14
        
        # Get embed dim (768 for base, 384 for small)
        self.embed_dim = self.encoder.embed_dim
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # Decoder head
        self.seg_head = DinoTinyDecoder(self.embed_dim, 1)
    
    def forward_features(self, x):
        """Extract spatial features from DINOv2."""
        B, C, H, W = x.shape
        
        with torch.no_grad():
            out = self.encoder.forward_features(x)
            feats = out['x_norm_patchtokens']  # (B, N, C) - excludes CLS token
        
        # Reshape to spatial
        B, N, C = feats.shape
        s = int(math.sqrt(N))
        fmap = feats.permute(0, 2, 1).reshape(B, C, s, s)
        
        return fmap
    
    def forward_seg(self, x):
        """Get segmentation logits."""
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (self.img_size, self.img_size))
    
    def forward(self, x):
        """Standard forward - returns logits."""
        return self.forward_seg(x)
    
    def predict(self, x):
        """Forward with sigmoid."""
        return torch.sigmoid(self.forward_seg(x))


def postprocess(preds, original_size, alpha_grad=0.35):
    """Postprocessing from working code - edge enhancement + morphology."""
    # Sobel edge detection
    gx = cv2.Sobel(preds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(preds, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    
    # Blend with edge info
    enhanced = (1 - alpha_grad) * preds + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Adaptive threshold
    thr = np.mean(enhanced) + 0.3 * np.std(enhanced)
    mask = (enhanced > thr).astype(np.uint8)
    
    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Resize to original
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask


@torch.no_grad()
def predict_with_tta(model, image):
    """Test-time augmentation with flips."""
    predictions = []
    
    # Original
    pred = torch.sigmoid(model.forward_seg(image))
    predictions.append(pred)
    
    # Horizontal flip
    pred = torch.sigmoid(model.forward_seg(torch.flip(image, dims=[3])))
    predictions.append(torch.flip(pred, dims=[3]))
    
    # Vertical flip
    pred = torch.sigmoid(model.forward_seg(torch.flip(image, dims=[2])))
    predictions.append(torch.flip(pred, dims=[2]))
    
    return torch.stack(predictions).mean(0)[0, 0].cpu().numpy()


class ForgeryLoss(nn.Module):
    """BCE + Dice loss for segmentation."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def create_model(size='base', img_size=512, device='cuda'):
    """
    Create model matching Kaggle winning architecture.
    
    Args:
        size: 'small' (faster) or 'base' (better, matches Kaggle)
        img_size: 512 recommended (matches Kaggle)
        device: cuda/mps/cpu
    """
    model_map = {
        'small': 'dinov2_vits14',  # 384 dim - faster
        'base': 'dinov2_vitb14',   # 768 dim - matches Kaggle
    }
    model = DinoSegmenter(model_map[size], img_size).to(device)
    loss_fn = ForgeryLoss()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {size}, Total params: {total:,}, Trainable: {trainable:,}")
    
    return model, loss_fn


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Use 'base' to match Kaggle exactly, 'small' for faster training
    model, loss_fn = create_model('small', img_size=512, device=device)
    
    # Test forward
    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
