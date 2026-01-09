"""
Training script for DinoSegmenter - matches Kaggle architecture.
Run this for 50+ epochs to get good results.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from dino_segmenter import create_model


class ForgeryDataset(Dataset):
    """Dataset that only loads forged images (they have masks)."""
    
    def __init__(self, images_dir, masks_dir, img_size=512):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        
        # Get forged images only (they have actual masks)
        forged_dir = self.images_dir / "forged"
        if forged_dir.exists():
            self.image_files = sorted(forged_dir.glob("*.png"))
        else:
            self.image_files = sorted(self.images_dir.glob("*.png"))
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image - normalize to [0,1]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        
        # Load mask
        mask_path = self.masks_dir / (img_path.stem + ".npy")
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)
            # Handle various mask shapes
            while mask.ndim > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]
            # Normalize to [0,1]
            if mask.max() > 1:
                mask = mask / 255.0
            # Resize
            mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
        else:
            mask = torch.zeros(self.img_size, self.img_size)
        
        mask = mask.unsqueeze(0)  # Add channel dim
        return image, mask


def compute_metrics(pred, target, threshold=0.5):
    """Compute IoU, F1, Precision, Recall."""
    pred_bin = (pred > threshold).float()
    
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall}


def train():
    # ============ CONFIG ============
    TRAIN_IMAGES = 'train_images'
    TRAIN_MASKS = 'train_masks'
    IMG_SIZE = 512
    BATCH_SIZE = 4  # Reduce if OOM
    EPOCHS = 30
    LR = 1e-4
    MODEL_SIZE = 'small'  # 'small' or 'base'
    # ================================
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: img_size={IMG_SIZE}, batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")
    
    # Dataset
    full_dataset = ForgeryDataset(TRAIN_IMAGES, TRAIN_MASKS, IMG_SIZE)
    
    # Split 85/15
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model, loss_fn = create_model(MODEL_SIZE, IMG_SIZE, device)
    
    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training
    best_iou = 0
    history = []
    
    print("\n" + "="*85)
    print(f"{'Ep':>3} | {'Train Loss':>10} | {'Val Loss':>10} | {'IoU':>7} | {'F1':>7} | {'Prec':>7} | {'Recall':>7}")
    print("="*85)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_losses = []
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        model.eval()
        val_losses = []
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = loss_fn(logits, masks)
                val_losses.append(loss.item())
                
                preds = torch.sigmoid(logits)
                all_metrics.append(compute_metrics(preds, masks))
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        # Save best
        marker = ""
        if avg_metrics['iou'] > best_iou:
            best_iou = avg_metrics['iou']
            torch.save(model.state_dict(), 'best_model.pth')
            marker = " *"
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **avg_metrics
        })
        
        print(f"{epoch+1:>3} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | {avg_metrics['iou']:>7.4f} | {avg_metrics['f1']:>7.4f} | {avg_metrics['precision']:>7.4f} | {avg_metrics['recall']:>7.4f}{marker}")
    
    # Summary
    print("="*85)
    print(f"\nBest IoU: {best_iou:.4f}")
    print(f"Model saved to best_model.pth")
    
    # Show improvement
    print(f"\nLoss: {history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f} ({100*(1-history[-1]['train_loss']/history[0]['train_loss']):.1f}% reduction)")
    print(f"IoU:  {history[0]['iou']:.4f} -> {history[-1]['iou']:.4f}")
    print(f"F1:   {history[0]['f1']:.4f} -> {history[-1]['f1']:.4f}")


if __name__ == '__main__':
    train()
