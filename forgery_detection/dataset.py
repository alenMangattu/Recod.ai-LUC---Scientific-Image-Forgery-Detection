"""
Dataset and Augmentation for Forgery Detection
===============================================
Heavy augmentation for competitive edge.
"""

import numpy as np
from PIL import Image
from PIL.Image import Resampling, Transpose
from pathlib import Path
import torch
from torch.utils.data import Dataset
import random


class ForgeryDataset(Dataset):
    """
    Dataset for loading forgery images and masks.
    
    Args:
        mask_files: List of mask file paths (.npy files)
        images_dir: Directory containing images (with 'forged' and 'authentic' subdirs)
        target_size: Size to resize images (512 recommended for competition)
        augment: Whether to apply augmentation
        training: Whether this is training mode
    """
    
    def __init__(self, mask_files, images_dir, target_size=512, augment=True, training=True):
        self.target_size = target_size
        self.augment = augment and training
        self.training = training
        
        images_dir = Path(images_dir)
        self.forged_dir = images_dir / 'forged'
        self.authentic_dir = images_dir / 'authentic'
        
        # Build valid pairs
        self.valid_pairs = []
        for mask_file in mask_files:
            mask_id = Path(mask_file).stem
            
            forged_path = self.forged_dir / f"{mask_id}.png"
            authentic_path = self.authentic_dir / f"{mask_id}.png"
            
            if forged_path.exists():
                self.valid_pairs.append((mask_file, forged_path))
            elif authentic_path.exists():
                self.valid_pairs.append((mask_file, authentic_path))
        
        print(f"Dataset: {len(self.valid_pairs)} valid pairs (augment={augment})")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        mask_file, image_path = self.valid_pairs[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask = np.load(mask_file)
        mask = self._process_mask(mask)
        mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Resize
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to tensors
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        
        # Normalize image with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensor format (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        mask = (mask > 0.5).float()  # Binarize
        
        return image, mask
    
    def _process_mask(self, mask):
        """Convert mask to 2D binary array."""
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[0]
        elif len(mask.shape) == 1:
            size = int(np.sqrt(mask.shape[0]))
            if size * size == mask.shape[0]:
                mask = mask.reshape(size, size)
        
        # Normalize to 0-1
        if mask.max() > 1:
            mask = mask.astype(float) / 255.0
        
        return (mask > 0.5).astype(float)
    
    def _augment(self, image, mask):
        """Apply heavy augmentation for training."""
        
        # Horizontal flip (50%)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Vertical flip (50%)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Rotation (90, 180, 270 degrees) - 30%
        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        
        # Random rotation (-30 to 30 degrees) - 30%
        if random.random() < 0.3:
            angle = random.uniform(-30, 30)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        
        # Color jitter (brightness, contrast, saturation) - 50%
        if random.random() < 0.5:
            image = self._color_jitter(image)
        
        # Random crop and resize - 30%
        if random.random() < 0.3:
            image, mask = self._random_crop(image, mask)
        
        # Gaussian noise - 20%
        if random.random() < 0.2:
            image = self._add_noise(image)
        
        return image, mask
    
    def _color_jitter(self, image):
        """Apply color jitter."""
        from PIL import ImageEnhance
        
        # Brightness
        factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Contrast
        factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(factor)
        
        # Saturation
        factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Color(image).enhance(factor)
        
        return image
    
    def _random_crop(self, image, mask):
        """Random crop and resize back."""
        w, h = image.size
        crop_ratio = random.uniform(0.7, 0.95)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        image = image.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))
        
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        
        return image, mask
    
    def _add_noise(self, image):
        """Add Gaussian noise."""
        img_array = np.array(image).astype(float)
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class TestDataset(Dataset):
    """Dataset for test images (no masks)."""
    
    def __init__(self, image_dir, target_size=512):
        self.target_size = target_size
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        print(f"Test dataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_id = image_path.stem
        
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, image_id, original_size


def get_dataloaders(train_masks_dir, train_images_dir, 
                    batch_size=8, target_size=512, 
                    val_split=0.15, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        train_masks_dir: Directory with mask .npy files
        train_images_dir: Directory with train images
        batch_size: Batch size
        target_size: Image size (512 recommended)
        val_split: Validation split ratio
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # Get all mask files
    mask_files = sorted(list(Path(train_masks_dir).glob('*.npy')))
    print(f"Found {len(mask_files)} mask files")
    
    # Split
    random.shuffle(mask_files)
    val_size = int(len(mask_files) * val_split)
    train_files = mask_files[val_size:]
    val_files = mask_files[:val_size]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_dataset = ForgeryDataset(
        train_files, train_images_dir, 
        target_size=target_size, augment=True, training=True
    )
    val_dataset = ForgeryDataset(
        val_files, train_images_dir,
        target_size=target_size, augment=False, training=False
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    print("Testing dataset...")
    # This would require actual data files to test
    print("Dataset module loaded successfully!")
