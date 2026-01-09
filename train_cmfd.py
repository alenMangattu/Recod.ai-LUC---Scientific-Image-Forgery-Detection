"""
Training script for Copy-Move Forgery Detection Model
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

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
            mask = Image.fromarray((mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8))
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        
        return image, mask


def train_epoch(model, loader, loss_fn, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = loss_fn(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    for images, masks in tqdm(loader, desc="Validating"):
        images = images.to(device)
        masks = masks.to(device)
        
        preds = model(images)
        loss = loss_fn(preds, masks)
        total_loss += loss.item()
        
        # IoU
        pred_bin = (preds > 0.5).float()
        intersection = (pred_bin * masks).sum()
        union = pred_bin.sum() + masks.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()
        total_iou += iou
    
    return total_loss / len(loader), total_iou / len(loader)


def train(
    train_images_dir,
    train_masks_dir,
    val_images_dir=None,
    val_masks_dir=None,
    epochs=50,
    batch_size=8,
    lr=1e-4,
    image_size=224,
    model_size='small',
    save_path='cmfd_best.pth',
    device='cuda',
):
    """Main training function."""
    
    # Create datasets
    train_dataset = CMFDDataset(train_images_dir, train_masks_dir, image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = None
    if val_images_dir and val_masks_dir:
        val_dataset = CMFDDataset(val_images_dir, val_masks_dir, image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model, loss_fn = create_model(model_size, freeze_backbone=True, device=device)
    print(f"Training on {len(train_dataset)} images")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss, val_iou = validate(model, val_loader, loss_fn, device)
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            metric = val_loss
        else:
            metric = train_loss
        
        if metric < best_loss:
            best_loss = metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, default='train_images')
    parser.add_argument('--train_masks', type=str, default='train_masks')
    parser.add_argument('--val_images', type=str, default=None)
    parser.add_argument('--val_masks', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--save_path', type=str, default='cmfd_best.pth')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train(
        args.train_images,
        args.train_masks,
        args.val_images,
        args.val_masks,
        args.epochs,
        args.batch_size,
        args.lr,
        args.image_size,
        args.model_size,
        args.save_path,
        device,
    )
