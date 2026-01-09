"""
Training script with detailed metrics tracking
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

from cmfd_model import create_model, CMFDLoss


class CMFDDataset(Dataset):
    """Dataset for copy-move forgery detection."""
    
    def __init__(self, images_dir, masks_dir, image_size=224):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        
        # Get image files
        self.image_files = []
        if (self.images_dir / "authentic").exists():
            self.image_files += list((self.images_dir / "authentic").glob("*.png"))
            self.image_files += list((self.images_dir / "forged").glob("*.png"))
        else:
            self.image_files = list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        # DINOv2 normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std
        
        # Load mask
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
    """Compute IoU, Precision, Recall, F1."""
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
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def run_training():
    """Main training loop with metrics tracking."""
    
    # Config
    TRAIN_IMAGES = 'train_images'
    TRAIN_MASKS = 'train_masks'
    EPOCHS = 20
    BATCH_SIZE = 4
    LR = 1e-4
    IMAGE_SIZE = 224
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Dataset
    dataset = CMFDDataset(TRAIN_IMAGES, TRAIN_MASKS, IMAGE_SIZE)
    print(f"Dataset size: {len(dataset)}")
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model, loss_fn = create_model('small', freeze_backbone=True, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_accuracy': [],
        'lr': [],
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                
                preds = model(images)
                loss = loss_fn(preds, masks)
                val_losses.append(loss.item())
                
                metrics = compute_metrics(preds, masks)
                all_metrics.append(metrics)
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_metrics['iou'])
        history['val_f1'].append(avg_metrics['f1'])
        history['val_precision'].append(avg_metrics['precision'])
        history['val_recall'].append(avg_metrics['recall'])
        history['val_accuracy'].append(avg_metrics['accuracy'])
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val IoU:    {avg_metrics['iou']:.4f}")
        print(f"  Val F1:     {avg_metrics['f1']:.4f}")
        print(f"  Val Prec:   {avg_metrics['precision']:.4f}")
        print(f"  Val Recall: {avg_metrics['recall']:.4f}")
        print(f"  Val Acc:    {avg_metrics['accuracy']:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'metrics': avg_metrics,
            }, 'cmfd_best.pth')
            print(f"  -> Saved best model!")
        
        print(f"{'='*60}\n")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print("\nLoss Progression:")
    print(f"  Start -> End")
    print(f"  Train: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print(f"  Val:   {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    
    print("\nMetrics Improvement:")
    for metric in ['iou', 'f1', 'precision', 'recall', 'accuracy']:
        key = f'val_{metric}'
        start = history[key][0]
        end = history[key][-1]
        best = max(history[key])
        print(f"  {metric.upper():10s}: {start:.4f} -> {end:.4f} (best: {best:.4f})")
    
    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nSaved training history to training_history.json")
    
    return history


if __name__ == '__main__':
    history = run_training()
