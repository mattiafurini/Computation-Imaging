"""
Forward Model: Degradation operators for Super-Resolution problem.

This module implements the degradation model: y = Hx + n
where:
  - x: Ground truth image (high resolution, 256x256)
  - H: Degradation operator (downsampling + blur)
  - n: Additive Gaussian noise
  - y: Observed degraded image
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path


class DegradationOperator:
    """
    Applies degradation (downsampling + noise) to an image.
    
    Parameters:
    -----------
    sr_factor : int
        Super-resolution upsampling factor (2 or 4)
    noise_std : float
        Standard deviation of Gaussian noise to add (0.005 or 0.01)
    seed : int
        Random seed for reproducibility
    device : str
        Device to use ('cpu' or 'cuda')
    """
    
    def __init__(self, sr_factor=2, noise_std=0.005, seed=42, device='cpu'):
        assert sr_factor in [2, 4], "sr_factor must be 2 or 4"
        assert noise_std in [0.005, 0.01], "noise_std must be 0.005 or 0.01"
        
        self.sr_factor = sr_factor
        self.noise_std = noise_std
        self.seed = seed
        self.device = device
        
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def __call__(self, img_tensor):
        """
        Apply degradation to image tensor.
        
        Parameters:
        -----------
        img_tensor : torch.Tensor or PIL.Image or numpy.ndarray
            Input image. If PIL/numpy, will be converted to tensor.
            Expected shape: (C, H, W) or (H, W, C) or (H, W)
        
        Returns:
        --------
        degraded : torch.Tensor
            Degraded image, same format as input (tensor)
        """
        
        # Convert input to tensor if needed
        if isinstance(img_tensor, Image.Image):
            img_tensor = torch.from_numpy(np.array(img_tensor)).float() / 255.0
            # Convert (H, W, C) -> (C, H, W)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        elif isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            if len(img_tensor.shape) == 3 and img_tensor.shape[2] in [3, 4]:
                img_tensor = img_tensor.permute(2, 0, 1)
        
        # Ensure tensor is float and on correct device
        img_tensor = img_tensor.float().to(self.device)
        
        # Add batch dimension if needed
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # Apply downsampling
        degraded = self._downsample(img_tensor)
        
        # Add Gaussian noise
        degraded = self._add_noise(degraded)
        
        # Remove batch dimension
        degraded = degraded.squeeze(0)
        
        return degraded
    
    def _downsample(self, x):
        """
        Downsample image using bicubic interpolation + anti-aliasing blur.
        
        This simulates a realistic degradation:
        1. Apply a blur kernel (Gaussian or box) to avoid aliasing
        2. Downsample by sr_factor
        
        Parameters:
        -----------
        x : torch.Tensor
            Image tensor of shape (B, C, H, W)
        
        Returns:
        --------
        downsampled : torch.Tensor
            Downsampled image of shape (B, C, H//sr_factor, W//sr_factor)
        """
        
        # Create anti-aliasing blur kernel
        # We use a Gaussian kernel to smooth before downsampling
        kernel_size = self.sr_factor * 2 + 1  # e.g., 5 for factor 2, 9 for factor 4
        sigma = self.sr_factor / np.sqrt(8 * np.log(2))  # FWHM = sr_factor
        
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(self.device)
        
        # Apply blur to each channel
        B, C, H, W = x.shape
        blurred = torch.zeros_like(x)
        for i in range(C):
            # Convolve with Gaussian kernel (kernel already has shape (1, 1, kernel_size, kernel_size))
            blurred[:, i:i+1, :, :] = F.conv2d(
                x[:, i:i+1, :, :], 
                kernel,
                padding=kernel_size // 2
            )
        
        # Downsample using F.interpolate with area mode
        downsampled = F.interpolate(
            blurred, 
            scale_factor=1/self.sr_factor, 
            mode='bilinear',
            align_corners=False
        )
        
        return downsampled
    
    def _add_noise(self, x):
        """
        Add Gaussian noise to image.
        
        Parameters:
        -----------
        x : torch.Tensor
            Image tensor of shape (B, C, H, W)
        
        Returns:
        --------
        noisy : torch.Tensor
            Noisy image with same shape
        """
        noise = torch.randn_like(x) * self.noise_std
        noisy = x + noise
        # Clip to valid range
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy
    
    @staticmethod
    def _create_gaussian_kernel(kernel_size, sigma):
        """
        Create 2D Gaussian kernel.
        
        Parameters:
        -----------
        kernel_size : int
            Size of the kernel (must be odd)
        sigma : float
            Standard deviation of Gaussian
        
        Returns:
        --------
        kernel : torch.Tensor
            2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size)
        """
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
        return kernel_2d.unsqueeze(0).unsqueeze(0)


def degrade_image(image_path, sr_factor=2, noise_std=0.005, seed=42, device='cpu'):
    """
    Convenience function: load image and apply degradation.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to input image
    sr_factor : int
        Super-resolution factor (2 or 4)
    noise_std : float
        Noise standard deviation (0.005 or 0.01)
    seed : int
        Random seed
    device : str
        Device ('cpu' or 'cuda')
    
    Returns:
    --------
    degraded : torch.Tensor
        Degraded image tensor in range [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    degrader = DegradationOperator(sr_factor=sr_factor, noise_std=noise_std, seed=seed, device=device)
    degraded = degrader(img)
    return degraded


def create_degraded_dataset(input_dir, output_dir, sr_factor=2, noise_std=0.005, seed=42, device='cpu'):
    """
    Create a degraded version of an entire dataset.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing ground truth images
    output_dir : str or Path
        Directory where degraded images will be saved
    sr_factor : int
        Super-resolution factor
    noise_std : float
        Noise level
    seed : int
        Random seed
    device : str
        Device to use
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    degrader = DegradationOperator(sr_factor=sr_factor, noise_std=noise_std, seed=seed, device=device)
    
    image_files = sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg'))
    
    print(f"Degrading {len(image_files)} images from {input_dir}")
    print(f"SR factor: {sr_factor}x, Noise std: {noise_std}")
    
    for idx, img_path in enumerate(image_files):
        degraded = degrade_image(img_path, sr_factor=sr_factor, noise_std=noise_std, seed=seed, device=device)
        
        # Convert back to PIL and save
        # degraded is in [0, 1] and shape (C, H, W)
        degraded_np = (degraded.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        degraded_img = Image.fromarray(degraded_np)
        
        output_path = output_dir / img_path.name
        degraded_img.save(output_path)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)}")
    
    print(f"Done! Degraded images saved to {output_dir}")


if __name__ == "__main__":
    # Example usage: degrade test set with 4x SR and noise 0.01
    test_dir = Path(__file__).parent.parent / "data" / "test"
    output_base = Path(__file__).parent.parent / "data"
    
    # Create degraded versions for all 4 configurations
    configs = [
        (2, 0.005, "test_sr2x_noise0005"),
        (2, 0.01, "test_sr2x_noise001"),
        (4, 0.005, "test_sr4x_noise0005"),
        (4, 0.01, "test_sr4x_noise001"),
    ]
    
    for sr, noise, output_name in configs:
        out_dir = output_base / output_name
        create_degraded_dataset(test_dir, out_dir, sr_factor=sr, noise_std=noise, seed=42)
