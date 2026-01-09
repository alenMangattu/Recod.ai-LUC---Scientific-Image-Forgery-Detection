"""
Download Pretrained Weights
============================
Run this script LOCALLY to download pretrained weights.
Then upload the weights folder to Kaggle as a dataset.

Usage:
    python download_weights.py

This will create a 'pretrained_weights' folder with:
    - resnet101_imagenet.pth
    - resnet50_imagenet.pth

Upload this folder to Kaggle as a dataset named 'pretrained-weights'
Then access from Kaggle at: /kaggle/input/pretrained-weights/
"""

import torch
import torchvision.models as models
from pathlib import Path


def download_weights(output_dir='pretrained_weights'):
    """Download and save pretrained weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING PRETRAINED WEIGHTS")
    print("=" * 60)
    
    # ResNet101 (ImageNet V2 weights - better than V1)
    print("\n1. Downloading ResNet101 (ImageNet)...")
    resnet101 = models.resnet101(weights='IMAGENET1K_V2')
    save_path = output_dir / 'resnet101_imagenet.pth'
    torch.save(resnet101.state_dict(), save_path)
    print(f"   Saved: {save_path}")
    print(f"   Size: {save_path.stat().st_size / 1e6:.1f} MB")
    
    # ResNet50 (lighter alternative)
    print("\n2. Downloading ResNet50 (ImageNet)...")
    resnet50 = models.resnet50(weights='IMAGENET1K_V2')
    save_path = output_dir / 'resnet50_imagenet.pth'
    torch.save(resnet50.state_dict(), save_path)
    print(f"   Saved: {save_path}")
    print(f"   Size: {save_path.stat().st_size / 1e6:.1f} MB")
    
    # EfficientNet-B4 (optional - very good performance)
    print("\n3. Downloading EfficientNet-B4 (ImageNet)...")
    try:
        efficientnet = models.efficientnet_b4(weights='IMAGENET1K_V1')
        save_path = output_dir / 'efficientnet_b4_imagenet.pth'
        torch.save(efficientnet.state_dict(), save_path)
        print(f"   Saved: {save_path}")
        print(f"   Size: {save_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"   Skipped (optional): {e}")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Go to Kaggle: https://www.kaggle.com/datasets/new")
    print("2. Upload the 'pretrained_weights' folder as a new dataset")
    print("3. Name it: 'pretrained-weights'")
    print("4. In your notebook, access via: /kaggle/input/pretrained-weights/")


if __name__ == '__main__':
    download_weights()
