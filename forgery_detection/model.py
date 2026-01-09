"""
Forgery Detection Model - ResNet101 U-Net with Pretrained Encoder
==================================================================
State-of-the-art segmentation model for scientific image forgery detection.
Loads pretrained ImageNet weights from local file (no internet required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, resnet50


class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""
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
    """Decoder block with skip connection and upsampling."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        # Attention gate for better feature selection
        self.attention = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_ch // 4, skip_ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Apply attention to skip connection
        skip = skip * self.attention(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FinalDecoder(nn.Module):
    """Final decoder block without skip connection."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet101UNet(nn.Module):
    """
    U-Net with pretrained ResNet101 encoder.
    
    Features:
    - Pretrained ImageNet weights (loaded from local file)
    - Attention gates on skip connections
    - Deep supervision for better gradient flow
    
    Args:
        weights_path: Path to pretrained ResNet101 weights (.pth file)
        freeze_encoder: Whether to freeze encoder layers initially
    """
    
    def __init__(self, weights_path=None, freeze_encoder=False):
        super().__init__()
        
        # Build ResNet101 backbone
        backbone = resnet101(weights=None)
        
        # Load pretrained weights if provided
        if weights_path is not None:
            print(f"Loading pretrained weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            backbone.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully!")
        else:
            print("WARNING: No pretrained weights provided. Training from scratch.")
        
        # Encoder layers (extract features at different scales)
        # Input: 3 channels -> Output sizes for 512x512 input:
        self.encoder1 = nn.Sequential(
            backbone.conv1,   # 3 -> 64, stride 2 (256x256)
            backbone.bn1,
            backbone.relu
        )
        self.pool = backbone.maxpool  # stride 2 (128x128)
        self.encoder2 = backbone.layer1  # 64 -> 256 (128x128)
        self.encoder3 = backbone.layer2  # 256 -> 512, stride 2 (64x64)
        self.encoder4 = backbone.layer3  # 512 -> 1024, stride 2 (32x32)
        self.encoder5 = backbone.layer4  # 1024 -> 2048, stride 2 (16x16)
        
        # Freeze encoder if requested (for initial fine-tuning)
        if freeze_encoder:
            for param in self.encoder1.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            # Keep later layers trainable
        
        # Decoder (with skip connections)
        self.decoder5 = DecoderBlock(2048, 1024, 512)   # 16->32
        self.decoder4 = DecoderBlock(512, 512, 256)     # 32->64
        self.decoder3 = DecoderBlock(256, 256, 128)     # 64->128
        self.decoder2 = DecoderBlock(128, 64, 64)       # 128->256
        self.decoder1 = FinalDecoder(64, 32)            # 256->512
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        # Deep supervision heads (auxiliary outputs for better training)
        self.deep4 = nn.Conv2d(512, 1, kernel_size=1)
        self.deep3 = nn.Conv2d(256, 1, kernel_size=1)
        
        # Initialize decoder weights
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder with Kaiming initialization."""
        for m in [self.decoder5, self.decoder4, self.decoder3, self.decoder2, 
                  self.decoder1, self.final_conv, self.deep4, self.deep3]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, return_deep=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_deep: If True, return deep supervision outputs for training
        
        Returns:
            If return_deep=False: (B, 1, H, W) sigmoid probabilities
            If return_deep=True: tuple of (main_output, deep4, deep3)
        """
        input_size = x.shape[2:]
        
        # Encoder
        e1 = self.encoder1(x)        # 64 channels, H/2
        e1_pooled = self.pool(e1)    # 64 channels, H/4
        e2 = self.encoder2(e1_pooled) # 256 channels, H/4
        e3 = self.encoder3(e2)       # 512 channels, H/8
        e4 = self.encoder4(e3)       # 1024 channels, H/16
        e5 = self.encoder5(e4)       # 2048 channels, H/32
        
        # Decoder with skip connections
        d5 = self.decoder5(e5, e4)   # 512 channels, H/16
        d4 = self.decoder4(d5, e3)   # 256 channels, H/8
        d3 = self.decoder3(d4, e2)   # 128 channels, H/4
        d2 = self.decoder2(d3, e1)   # 64 channels, H/2
        d1 = self.decoder1(d2)       # 32 channels, H
        
        # Final output
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        if return_deep:
            # Deep supervision outputs
            deep4 = self.deep4(d5)
            deep4 = F.interpolate(deep4, size=input_size, mode='bilinear', align_corners=False)
            deep3 = self.deep3(d4)
            deep3 = F.interpolate(deep3, size=input_size, mode='bilinear', align_corners=False)
            return torch.sigmoid(out), torch.sigmoid(deep4), torch.sigmoid(deep3)
        
        return torch.sigmoid(out)


class ResNet50UNet(nn.Module):
    """
    Lighter version with ResNet50 encoder.
    Use this if you have memory constraints.
    """
    
    def __init__(self, weights_path=None, freeze_encoder=False):
        super().__init__()
        
        backbone = resnet50(weights=None)
        
        if weights_path is not None:
            print(f"Loading pretrained weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            backbone.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully!")
        
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4
        
        if freeze_encoder:
            for param in self.encoder1.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
        
        self.decoder5 = DecoderBlock(2048, 1024, 512)
        self.decoder4 = DecoderBlock(512, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = FinalDecoder(64, 32)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    
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


# Alias for compatibility
UNetSegmentation = ResNet101UNet
DeepLabV3Segmentation = ResNet101UNet


def get_model(model_name='resnet101', weights_path=None, freeze_encoder=False):
    """
    Factory function to get model by name.
    
    Args:
        model_name: 'resnet101' or 'resnet50'
        weights_path: Path to pretrained weights
        freeze_encoder: Whether to freeze encoder layers
    
    Returns:
        Model instance
    """
    if model_name == 'resnet101':
        return ResNet101UNet(weights_path=weights_path, freeze_encoder=freeze_encoder)
    elif model_name == 'resnet50':
        return ResNet50UNet(weights_path=weights_path, freeze_encoder=freeze_encoder)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    # Test model
    print("Testing ResNet101UNet...")
    model = ResNet101UNet(weights_path=None)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Test passed!")
