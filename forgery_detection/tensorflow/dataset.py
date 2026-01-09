"""
Dataset and Data Loading for Forgery Detection - TensorFlow/Keras
==================================================================
Uses tf.data.Dataset for efficient data loading with PIL-based augmentation.

The augmentation strategy is heavy and specifically designed for
copy-move forgery detection.
"""

import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import random
import re


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ForgeryDataset:
    """
    Dataset class for forgery detection.
    Uses PIL for augmentation and converts to TensorFlow tensors.
    
    Args:
        mask_files: List of mask file paths (.npy files)
        images_dir: Directory containing images (with 'forged' and 'authentic' subdirs)
        target_size: Size to resize images
        augment: Whether to apply augmentation
        training: Whether this is for training
    """
    
    def __init__(
        self,
        items,
        target_size=384,
        augment=True,
        training=True,
    ):
        self.target_size = target_size
        self.augment = augment and training
        self.training = training

        # items: list of (case_id, image_path, mask_files[list[str]])
        self.items = [(cid, str(img), [str(m) for m in masks]) for cid, img, masks in items]
        print(f"ForgeryDataset: {len(self.items)} cases (augment={augment})")
    
    def __len__(self):
        return len(self.items)
    
    def load_sample(self, idx):
        """Load a single sample (used by tf.data.Dataset.from_generator)."""
        case_id, image_path, mask_files = self.items[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load + union masks for this case (some cases have multiple duplicated regions)
        mask_union = None
        for mf in mask_files:
            m = np.load(mf)
            m = self._process_mask(m)
            if mask_union is None:
                mask_union = m
            else:
                mask_union = np.maximum(mask_union, m)
        if mask_union is None:
            # Shouldn't happen, but keep shape stable
            mask_union = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        mask = Image.fromarray((mask_union * 255).astype(np.uint8), mode='L')
        
        # Resize
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
        
        return image, mask
    
    def _process_mask(self, mask):
        """Convert mask to 2D binary array."""
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[:, :, 0]
        
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
        
        # Rotation 90/180/270 (30%)
        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        
        # Random rotation -25 to 25 degrees (30%)
        if random.random() < 0.3:
            angle = random.uniform(-25, 25)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        
        # Color jitter (50%)
        if random.random() < 0.5:
            image = self._color_jitter(image)
        
        # Gaussian blur (20%)
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Random crop and resize (30%)
        if random.random() < 0.3:
            image, mask = self._random_crop(image, mask)
        
        return image, mask
    
    def _color_jitter(self, image):
        """Apply color jitter augmentation."""
        # Brightness
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        # Contrast
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        # Saturation
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image
    
    def _random_crop(self, image, mask):
        """Random crop and resize back."""
        w, h = image.size
        crop_ratio = random.uniform(0.75, 0.95)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        image = image.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))
        
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        
        return image, mask
    
    def generator(self):
        """Generator function for tf.data.Dataset."""
        indices = list(range(len(self.pairs)))
        if self.training:
            random.shuffle(indices)
        
        for idx in indices:
            yield self.load_sample(idx)


def create_tf_dataset(dataset, batch_size, shuffle_buffer=1000):
    """
    Create a tf.data.Dataset from ForgeryDataset.
    
    Args:
        dataset: ForgeryDataset instance
        batch_size: Batch size
        shuffle_buffer: Buffer size for shuffling (only for training)
    
    Returns:
        tf.data.Dataset
    """
    target_size = dataset.target_size
    
    tf_dataset = tf.data.Dataset.from_generator(
        dataset.generator,
        output_signature=(
            tf.TensorSpec(shape=(target_size, target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(target_size, target_size, 1), dtype=tf.float32),
        )
    )
    
    if dataset.training:
        tf_dataset = tf_dataset.shuffle(shuffle_buffer)
    
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=dataset.training)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset


def create_datasets(train_masks_dir, train_images_dir, target_size=384, val_split=0.1):
    """
    Create training and validation datasets.
    
    Args:
        train_masks_dir: Directory with mask .npy files
        train_images_dir: Directory with training images
        target_size: Image size
        val_split: Validation split ratio
    
    Returns:
        train_dataset, val_dataset (ForgeryDataset instances)
    """
    return create_datasets_with_supplemental(
        train_masks_dir=train_masks_dir,
        train_images_dir=train_images_dir,
        target_size=target_size,
        val_split=val_split,
        supplemental_masks_dir=None,
        supplemental_images_dir=None,
    )


_MASK_SUFFIX_RE = re.compile(r"^(?P<case>.+?)_(?P<idx>\d+)$")


def _mask_path_to_case_id(mask_path: str) -> str:
    stem = Path(mask_path).stem
    m = _MASK_SUFFIX_RE.match(stem)
    if m:
        return m.group("case")
    return stem


def _resolve_image_path(case_id: str, train_images_dir: str, supplemental_images_dir: str | None):
    candidates = []
    train_dir = Path(train_images_dir)

    # Common layouts
    candidates.append(train_dir / f"{case_id}.png")
    candidates.append(train_dir / "forged" / f"{case_id}.png")
    candidates.append(train_dir / "authentic" / f"{case_id}.png")

    if supplemental_images_dir:
        sup_dir = Path(supplemental_images_dir)
        candidates.append(sup_dir / f"{case_id}.png")
        candidates.append(sup_dir / "forged" / f"{case_id}.png")
        candidates.append(sup_dir / "authentic" / f"{case_id}.png")

    for p in candidates:
        if p.exists():
            return p
    return None


def _build_items_from_mask_dirs(
    train_masks_dir: str,
    train_images_dir: str,
    supplemental_masks_dir: str | None,
    supplemental_images_dir: str | None,
):
    mask_paths = sorted([str(p) for p in Path(train_masks_dir).glob("*.npy")])
    if supplemental_masks_dir:
        mask_paths += sorted([str(p) for p in Path(supplemental_masks_dir).glob("*.npy")])

    print(f"Found {len(mask_paths)} mask files (train+supplemental)")

    # Group by case_id
    by_case: dict[str, list[str]] = {}
    for mp in mask_paths:
        cid = _mask_path_to_case_id(mp)
        by_case.setdefault(cid, []).append(mp)

    case_ids = sorted(by_case.keys())
    items = []
    missing_images = 0
    for cid in case_ids:
        img_path = _resolve_image_path(cid, train_images_dir, supplemental_images_dir)
        if img_path is None:
            missing_images += 1
            continue
        items.append((cid, img_path, by_case[cid]))

    if missing_images:
        print(f"[WARN] {missing_images} cases skipped (no matching image found).")
    print(f"Usable cases: {len(items)}")
    return items


def create_datasets_with_supplemental(
    train_masks_dir: str,
    train_images_dir: str,
    target_size=384,
    val_split=0.1,
    supplemental_masks_dir: str | None = None,
    supplemental_images_dir: str | None = None,
):
    """
    Create train/val datasets by splitting on case_id (not mask file), and union
    multiple masks per case.
    """
    items = _build_items_from_mask_dirs(
        train_masks_dir=train_masks_dir,
        train_images_dir=train_images_dir,
        supplemental_masks_dir=supplemental_masks_dir,
        supplemental_images_dir=supplemental_images_dir,
    )

    case_ids = [cid for cid, _, _ in items]
    random.shuffle(case_ids)
    val_size = int(len(case_ids) * val_split)
    val_set = set(case_ids[:val_size])

    train_items = [it for it in items if it[0] not in val_set]
    val_items = [it for it in items if it[0] in val_set]

    print(f"Train cases: {len(train_items)}, Val cases: {len(val_items)}")

    train_dataset = ForgeryDataset(
        train_items,
        target_size=target_size,
        augment=True,
        training=True,
    )
    val_dataset = ForgeryDataset(
        val_items,
        target_size=target_size,
        augment=False,
        training=False,
    )

    return train_dataset, val_dataset


def create_dataloaders(train_masks_dir, train_images_dir, batch_size=4, 
                       target_size=384, val_split=0.1):
    """
    Create training and validation dataloaders (tf.data.Dataset).
    
    Args:
        train_masks_dir: Directory with mask .npy files
        train_images_dir: Directory with training images
        batch_size: Batch size
        target_size: Image size
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader (tf.data.Dataset instances)
    """
    train_dataset, val_dataset = create_datasets(
        train_masks_dir, train_images_dir, 
        target_size=target_size, val_split=val_split
    )
    
    train_loader = create_tf_dataset(train_dataset, batch_size)
    val_loader = create_tf_dataset(val_dataset, batch_size)
    
    return train_loader, val_loader


class TestDataset:
    """Dataset for test images (no masks)."""
    
    def __init__(self, image_dir, target_size=384):
        self.target_size = target_size
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        print(f"TestDataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def load_image(self, idx):
        """Load a single test image."""
        image_path = self.image_files[idx]
        image_id = image_path.stem
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        
        return image, image_id, original_size
    
    def load_image_for_tta(self, idx):
        """Load image for TTA prediction."""
        image_path = self.image_files[idx]
        image_id = image_path.stem
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        image_resized = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        
        return image_np, image_id, original_size


if __name__ == '__main__':
    print("Dataset module loaded successfully!")
    print("To test with real data, use create_dataloaders() with actual paths.")
