"""
Training Script for UNet Super-Resolution.

Trains a separate UNet model for each degradation configuration
(SR factor × noise level). Each model learns the mapping:
    bicubic_upsample(degraded) → ground_truth

Usage:
    python train_unet.py --sr_factor 2 --noise_std 0.005 --epochs 50
    python train_unet.py --sr_factor 4 --noise_std 0.01 --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "methods"))

from methods.unet import UNet, count_parameters
from degradations import DegradationOperator
from normalization import default_normalizer
from metrics import compute_psnr, compute_ssim


class SRDataset(Dataset):
    """
    PyTorch Dataset for Super-Resolution training.

    Loads ground-truth images from disk, applies degradation on-the-fly,
    and returns (bicubic_upsampled_degraded, ground_truth) pairs.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing ground-truth 256×256 images.
    sr_factor : int
        Super-resolution factor (2 or 4).
    noise_std : float
        Noise standard deviation (0.005 or 0.01).
    max_images : int, optional
        Maximum number of images to use (for debugging).
    """

    def __init__(self, image_dir, sr_factor=2, noise_std=0.005, max_images=None):
        self.image_dir = Path(image_dir)
        self.sr_factor = sr_factor
        self.noise_std = noise_std

        self.files = sorted(self.image_dir.glob("*.png"))
        if max_images is not None:
            self.files = self.files[:max_images]

        self.degrader = DegradationOperator(
            sr_factor=sr_factor,
            noise_std=noise_std,
            seed=42
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns
        -------
        tuple of (input, target)
            input : torch.Tensor, shape (3, 256, 256) — bicubic upsampled degraded
            target : torch.Tensor, shape (3, 256, 256) — ground truth
        """
        # Load ground truth
        img = Image.open(self.files[idx]).convert('RGB')
        gt = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        gt = default_normalizer.normalize(gt)  # [0, 1]

        # Apply degradation (produces low-res image)
        # Reset seed per-sample for varied noise
        self.degrader.seed = 42 + idx
        torch.manual_seed(42 + idx)
        degraded = self.degrader(img)  # (C, H_lr, W_lr)

        # Bicubic upsample back to 256x256 (this is the UNet input)
        degraded_up = F.interpolate(
            degraded.unsqueeze(0),
            size=(256, 256),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        degraded_up = torch.clamp(degraded_up, 0.0, 1.0)

        return degraded_up, gt


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation. Returns average loss, mean PSNR, mean SSIM."""
    model.eval()
    total_loss = 0.0
    psnr_values = []
    ssim_values = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Compute per-image metrics
        for i in range(outputs.shape[0]):
            psnr_values.append(compute_psnr(outputs[i], targets[i]))
            ssim_values.append(compute_ssim(outputs[i], targets[i]))

    avg_loss = total_loss / len(dataloader)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_loss, avg_psnr, avg_ssim


def main(args):
    """Main training loop."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    model_name = f"unet_sr{args.sr_factor}x_noise{str(args.noise_std).replace('.', '')}"
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"SR Factor: {args.sr_factor}x | Noise: {args.noise_std}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    # Datasets
    train_dataset = SRDataset(data_dir / "train", args.sr_factor, args.noise_std)
    val_dataset = SRDataset(data_dir / "val", args.sr_factor, args.noise_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # Model
    model = UNet(in_channels=3, out_channels=3).to(device)
    print(f"Parameters: {count_parameters(model):,}\n")

    # Training setup
    criterion = nn.L1Loss()  # L1 preserves details better than MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=5)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': [],
    }
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

        # LR scheduling
        scheduler.step(val_loss)

        # Log
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"PSNR: {val_psnr:.2f} dB | "
              f"SSIM: {val_ssim:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(float(val_psnr))
        history['val_ssim'].append(float(val_ssim))

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = weights_dir / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'sr_factor': args.sr_factor,
                'noise_std': args.noise_std,
            }, save_path)
            print(f"  → Saved best model (PSNR: {val_psnr:.2f} dB)")

    # Save final model
    final_path = weights_dir / f"{model_name}_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'sr_factor': args.sr_factor,
        'noise_std': args.noise_std,
    }, final_path)

    # Save training history
    history_path = weights_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"Best model:  {weights_dir / f'{model_name}_best.pth'}")
    print(f"Final model: {final_path}")
    print(f"History:     {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for Super-Resolution")
    parser.add_argument("--sr_factor", type=int, default=2, choices=[2, 4],
                        help="Super-resolution factor (default: 2)")
    parser.add_argument("--noise_std", type=float, default=0.005,
                        help="Noise standard deviation (default: 0.005)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers (default: 4)")
    args = parser.parse_args()
    main(args)
