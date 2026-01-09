"""
Simplified CMFD Model - matches the working architecture
DINOv2 encoder + simple decoder head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleDecoder(nn.Module):
    """Simple 3-layer decoder - proven to work."""
    def __init__(self, in_ch=384, out_ch=1):  # 384 for ViT-S, 768 for ViT-B
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, features, target_size):
        # Upsample features to target size, then decode
        x = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
        return self.net(x)


class CMFDSimple(nn.Module):
    """
    Simple Copy-Move Forgery Detection model.
    DINOv2 backbone (frozen) + simple conv decoder.
    """
    def __init__(self, model_name='dinov2_vits14', freeze_backbone=True):
        super().__init__()
        
        # Load DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.patch_size = 14
        self.embed_dim = self.backbone.embed_dim  # 384 for vits, 768 for vitb
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # Simple decoder
        self.decoder = SimpleDecoder(in_ch=self.embed_dim, out_ch=1)
    
    def extract_features(self, x):
        """Extract spatial feature map from DINOv2."""
        B, C, H, W = x.shape
        
        # Get patch tokens
        with torch.no_grad():
            out = self.backbone.forward_features(x)
            tokens = out['x_norm_patchtokens']  # (B, N, C) excluding CLS
        
        # Reshape to spatial
        h = H // self.patch_size
        w = W // self.patch_size
        features = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        
        return features
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image (should be normalized with ImageNet stats)
        Returns:
            logits: (B, 1, H, W) raw logits (apply sigmoid for probabilities)
        """
        B, C, H, W = x.shape
        features = self.extract_features(x)
        logits = self.decoder(features, (H, W))
        return logits
    
    def predict(self, x):
        """Forward with sigmoid."""
        return torch.sigmoid(self.forward(x))


class CMFDLoss(nn.Module):
    """BCE + Dice loss."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def create_model(size='small', freeze_backbone=True, device='cuda'):
    """Create model and loss function."""
    model_map = {
        'small': 'dinov2_vits14',  # 384 dim, 22M params
        'base': 'dinov2_vitb14',   # 768 dim, 86M params
    }
    model = CMFDSimple(model_map[size], freeze_backbone).to(device)
    loss_fn = CMFDLoss()
    return model, loss_fn


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model, loss_fn = create_model('small', device=device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Test
    x = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
