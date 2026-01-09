"""
Proper training run with enough data and epochs to show improvement
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from PIL import Image
import numpy as np

from cmfd_model import create_model, CMFDLoss


class CMFDDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=224):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        
        self.image_files = []
        if (self.images_dir / "authentic").exists():
            self.image_files += list((self.images_dir / "authentic").glob("*.png"))
            self.image_files += list((self.images_dir / "forged").glob("*.png"))
        else:
            self.image_files = list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std
        
        mask_path = self.masks_dir / (img_path.stem + ".npy")
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)
            while mask.ndim > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]
            if mask.max() <= 1.0:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)
            mask = Image.fromarray(mask_uint8)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        
        return image, mask


def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


def main():
    # Config
    SUBSET_SIZE = 500  # More data
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 3e-4
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: {SUBSET_SIZE} images, {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    
    # Dataset
    full_dataset = CMFDDataset('train_images', 'train_masks', 224)
    print(f"Full dataset: {len(full_dataset)} images")
    
    # Use subset - sample forged images which have actual masks
    # Filter to only include forged images (they have non-zero masks)
    forged_indices = []
    for i, f in enumerate(full_dataset.image_files):
        if 'forged' in str(f):
            forged_indices.append(i)
    
    # Take subset of forged images
    indices = forged_indices[:min(SUBSET_SIZE, len(forged_indices))]
    dataset = Subset(full_dataset, indices)
    print(f"Using {len(dataset)} forged images (have actual masks)")
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model, loss_fn = create_model('small', freeze_backbone=True, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}\n")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Tracking
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [], 'val_prec': [], 'val_rec': []}
    
    print("="*80)
    print(f"{'Ep':>3} | {'Train Loss':>10} | {'Val Loss':>10} | {'IoU':>7} | {'F1':>7} | {'Prec':>7} | {'Recall':>7}")
    print("="*80)
    
    best_iou = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step()
        
        # Validate
        model.eval()
        val_losses = []
        all_metrics = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = loss_fn(preds, masks)
                val_losses.append(loss.item())
                all_metrics.append(compute_metrics(preds, masks))
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        # Record
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_metrics['iou'])
        history['val_f1'].append(avg_metrics['f1'])
        history['val_prec'].append(avg_metrics['precision'])
        history['val_rec'].append(avg_metrics['recall'])
        
        # Save best
        marker = ""
        if avg_metrics['iou'] > best_iou:
            best_iou = avg_metrics['iou']
            torch.save(model.state_dict(), 'cmfd_best.pth')
            marker = " *"
        
        print(f"{epoch+1:>3} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | {avg_metrics['iou']:>7.4f} | {avg_metrics['f1']:>7.4f} | {avg_metrics['precision']:>7.4f} | {avg_metrics['recall']:>7.4f}{marker}")
    
    # Summary
    print("="*80)
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    print(f"\n📉 LOSS PROGRESSION:")
    print(f"   Train Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}  ({100*(1-history['train_loss'][-1]/history['train_loss'][0]):+.1f}%)")
    print(f"   Val Loss:   {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}  ({100*(1-history['val_loss'][-1]/history['val_loss'][0]):+.1f}%)")
    
    print(f"\n📈 METRICS IMPROVEMENT:")
    print(f"   IoU:       {history['val_iou'][0]:.4f} → {history['val_iou'][-1]:.4f}  (best: {max(history['val_iou']):.4f})")
    print(f"   F1:        {history['val_f1'][0]:.4f} → {history['val_f1'][-1]:.4f}  (best: {max(history['val_f1']):.4f})")
    print(f"   Precision: {history['val_prec'][0]:.4f} → {history['val_prec'][-1]:.4f}  (best: {max(history['val_prec']):.4f})")
    print(f"   Recall:    {history['val_rec'][0]:.4f} → {history['val_rec'][-1]:.4f}  (best: {max(history['val_rec']):.4f})")
    
    print(f"\n✓ Best model saved to cmfd_best.pth (IoU: {best_iou:.4f})")
    print("="*80)
    
    return history


if __name__ == '__main__':
    main()
