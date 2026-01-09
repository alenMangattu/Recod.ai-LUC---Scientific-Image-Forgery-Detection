"""
Loss Functions for Forgery Detection
=====================================
Combined losses for better segmentation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary loss - penalizes errors at mask boundaries.
    Critical for forgery detection where edges matter.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def get_boundary(self, mask):
        """Extract boundary from mask using Sobel operator."""
        device = mask.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Apply Sobel filters
        edge_x = F.conv2d(mask, sobel_x, padding=1)
        edge_y = F.conv2d(mask, sobel_y, padding=1)
        
        # Magnitude
        boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        boundary = (boundary > 0.1).float()
        
        return boundary
    
    def forward(self, pred, target):
        # Get boundaries
        pred_boundary = self.get_boundary(pred)
        target_boundary = self.get_boundary(target)
        
        # Dice loss on boundaries
        smooth = 1.0
        pred_flat = pred_boundary.view(-1)
        target_flat = target_boundary.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss for forgery detection:
    - Dice: Overall region similarity
    - Focal: Handle class imbalance (forgery pixels are rare)
    - Boundary: Edge accuracy (important for localization)
    - BCE: Standard pixel-wise loss
    
    Weights are tuned for forgery detection task.
    """
    
    def __init__(self, dice_weight=1.0, focal_weight=1.0, 
                 boundary_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred, target):
        loss = 0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        if self.boundary_weight > 0:
            loss += self.boundary_weight * self.boundary_loss(pred, target)
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(pred, target)
        
        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Loss with deep supervision - uses auxiliary outputs.
    """
    
    def __init__(self, main_weight=1.0, aux_weights=(0.4, 0.2)):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weights = aux_weights
        self.loss_fn = CombinedLoss()
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: tuple of (main_output, aux1, aux2) or just main_output
            target: ground truth mask
        """
        if isinstance(outputs, tuple):
            main_out, aux1, aux2 = outputs
            loss = self.main_weight * self.loss_fn(main_out, target)
            loss += self.aux_weights[0] * self.loss_fn(aux1, target)
            loss += self.aux_weights[1] * self.loss_fn(aux2, target)
        else:
            loss = self.loss_fn(outputs, target)
        
        return loss


def get_loss(loss_name='combined', **kwargs):
    """Factory function for loss functions."""
    if loss_name == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'deep_supervision':
        return DeepSupervisionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


if __name__ == '__main__':
    # Test losses
    print("Testing losses...")
    pred = torch.rand(2, 1, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.5).float()
    
    for name, loss_fn in [('dice', DiceLoss()), 
                          ('focal', FocalLoss()),
                          ('boundary', BoundaryLoss()),
                          ('combined', CombinedLoss())]:
        loss = loss_fn(pred, target)
        print(f"{name}: {loss.item():.4f}")
    
    print("Loss functions OK!")
