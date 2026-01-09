"""
Copy-Move Forgery Detection Model
Uses DINOv2 for feature extraction + self-correlation + U-Net segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCorrelation(nn.Module):
    """Compute self-correlation map from feature tensor."""
    
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature tensor
        Returns:
            corr: (B, H*W, H, W) correlation map - for each position, similarity to all others
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: (B, C, H*W)
        x_flat = x.reshape(B, C, -1)
        
        if self.normalize:
            x_flat = F.normalize(x_flat, dim=1)
        
        # Self-correlation: (B, H*W, H*W)
        corr = torch.bmm(x_flat.transpose(1, 2).contiguous(), x_flat)
        
        # Reshape to (B, H*W, H, W) for spatial interpretation
        corr = corr.reshape(B, H * W, H, W)
        
        return corr


class CorrelationPooling(nn.Module):
    """Pool correlation maps to get top-k similarity channels."""
    
    def __init__(self, top_k=16):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, corr):
        """
        Args:
            corr: (B, H*W, H, W) full correlation
        Returns:
            pooled: (B, top_k, H, W) top-k correlation responses per position
        """
        B, N, H, W = corr.shape
        
        # For each spatial position, get top-k correlations (excluding self)
        # Reshape to (B, H, W, H*W)
        corr_reshaped = corr.permute(0, 2, 3, 1).contiguous()
        
        # Get top-k values (excluding the max which is self-correlation)
        topk_vals, _ = torch.topk(corr_reshaped, k=self.top_k + 1, dim=-1)
        
        # Skip first (self) and take next top_k
        topk_vals = topk_vals[..., 1:self.top_k + 1].contiguous()
        
        # Back to (B, top_k, H, W)
        pooled = topk_vals.permute(0, 3, 1, 2).contiguous()
        
        return pooled


class ConvBlock(nn.Module):
    """Simple conv block with BatchNorm and ReLU."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class SegmentationHead(nn.Module):
    """Simple U-Net style decoder for segmentation."""
    
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        
        # Encoder (lightweight, main features come from DINOv2)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)


class CMFDModel(nn.Module):
    """
    Copy-Move Forgery Detection Model
    
    Architecture:
    1. DINOv2 backbone extracts rich features
    2. Self-correlation finds similar patches
    3. Correlation pooling selects top-k matches
    4. Segmentation head outputs binary mask
    """
    
    def __init__(self, 
                 dino_model='dinov2_vits14',
                 correlation_top_k=16,
                 seg_base_channels=64,
                 freeze_backbone=True):
        super().__init__()
        
        # Load DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', dino_model)
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from model
        self.feat_dim = self.backbone.embed_dim
        
        # Self-correlation module
        self.self_corr = SelfCorrelation(normalize=True)
        self.corr_pool = CorrelationPooling(top_k=correlation_top_k)
        
        # Project correlation features
        self.corr_proj = nn.Sequential(
            nn.Conv2d(correlation_top_k, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Segmentation head: RGB (3) + correlation features (32)
        self.seg_head = SegmentationHead(
            in_channels=3 + 32,
            base_channels=seg_base_channels
        )
    
    def extract_features(self, x):
        """Extract patch features from DINOv2."""
        B, C, H, W = x.shape
        
        # DINOv2 expects specific input size (multiple of patch_size)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Get patch tokens (excluding CLS token)
        with torch.no_grad() if not self.training or not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # (B, N, C)
        
        # Reshape to spatial: (B, C, H, W)
        patch_tokens = patch_tokens.transpose(1, 2)  # (B, C, N)
        patch_tokens = patch_tokens.view(B, self.feat_dim, h_patches, w_patches)
        
        return patch_tokens
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image, should be multiple of 14
        Returns:
            mask: (B, 1, H, W) segmentation mask
        """
        B, C, H, W = x.shape
        
        # Extract DINOv2 features
        features = self.extract_features(x)  # (B, feat_dim, H/14, W/14)
        
        # Compute self-correlation
        corr = self.self_corr(features)  # (B, N, H/14, W/14)
        
        # Pool to top-k correlations
        corr_pooled = self.corr_pool(corr)  # (B, top_k, H/14, W/14)
        
        # Project and upsample correlation features to input resolution
        corr_features = self.corr_proj(corr_pooled)  # (B, 32, H/14, W/14)
        corr_features = F.interpolate(corr_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate with input image
        seg_input = torch.cat([x, corr_features], dim=1)  # (B, 35, H, W)
        
        # Segmentation
        mask = self.seg_head(seg_input)
        
        return torch.sigmoid(mask)
    
    def get_correlation_map(self, x):
        """Get correlation map for visualization."""
        features = self.extract_features(x)
        corr = self.self_corr(features)
        corr_pooled = self.corr_pool(corr)
        return corr_pooled


# Loss functions
class CMFDLoss(nn.Module):
    """Combined loss for copy-move forgery detection."""
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def create_model(model_size='small', freeze_backbone=True, device='cuda'):
    """
    Create CMFD model.
    
    Args:
        model_size: 'small' (ViT-S), 'base' (ViT-B), or 'large' (ViT-L)
        freeze_backbone: whether to freeze DINOv2 weights
        device: target device
    
    Returns:
        model, loss_fn
    """
    model_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14', 
        'large': 'dinov2_vitl14',
    }
    
    model = CMFDModel(
        dino_model=model_map[model_size],
        freeze_backbone=freeze_backbone,
    ).to(device)
    
    loss_fn = CMFDLoss()
    
    return model, loss_fn


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, loss_fn = create_model('small', device=device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    target = torch.rand(2, 1, 224, 224).to(device)
    
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    loss = loss_fn(out, target)
    print(f"Loss: {loss.item():.4f}")
