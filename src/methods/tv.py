"""
Total Variation (TV) Regularization for Super-Resolution.

Solves the inverse problem:
    x* = argmin_x  ½||Hx - y||² + λ·TV(x)

where:
    - H: degradation operator (blur + downsampling)
    - y: observed low-resolution noisy image
    - TV(x): anisotropic Total Variation prior
    - λ: regularization parameter

Uses FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for acceleration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from normalization import default_normalizer
except ImportError:
    from src.normalization import default_normalizer


class TVSuperResolution:
    """
    Super-Resolution via Total Variation regularization.

    Reconstructs a high-resolution image from a degraded low-resolution
    observation by minimizing a data-fidelity term plus a TV prior.

    Parameters
    ----------
    sr_factor : int
        Super-resolution factor (2 or 4).
    noise_std : float
        Noise level of the degraded observation (0.005 or 0.01).
    lambda_tv : float
        Regularization parameter controlling the strength of the TV prior.
        Higher values produce smoother results, lower values preserve more detail.
    n_iter : int
        Number of FISTA iterations.
    step_size : float
        Gradient descent step size (must be < 1/L where L is the Lipschitz
        constant of the data-fidelity gradient).
    device : str
        Device to run on ('cpu' or 'cuda').
    """

    def __init__(self, sr_factor=2, noise_std=0.005, lambda_tv=0.01,
                 n_iter=500, step_size=0.005, device='cpu'):
        assert sr_factor in [2, 4], "sr_factor must be 2 or 4"
        self.sr_factor = sr_factor
        self.noise_std = noise_std
        self.lambda_tv = lambda_tv
        self.n_iter = n_iter
        self.step_size = step_size
        self.device = device

        # Pre-compute the Gaussian blur kernel (same as in degradations.py)
        kernel_size = sr_factor * 2 + 1
        sigma = sr_factor / np.sqrt(8 * np.log(2))
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def solve(self, degraded_img):
        """
        Reconstruct a high-resolution image from a degraded observation.

        Parameters
        ----------
        degraded_img : torch.Tensor or PIL.Image or numpy.ndarray
            Degraded low-resolution image.
            Expected shape: (C, H_lr, W_lr) where H_lr = 256/sr_factor.

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (C, 256, 256) in range [0, 1].
        """
        # Convert input to tensor
        y = self._prepare_input(degraded_img)  # (1, C, H_lr, W_lr)
        C = y.shape[1]
        H_hr = y.shape[2] * self.sr_factor
        W_hr = y.shape[3] * self.sr_factor

        # Initialize with bicubic upsampling of the degraded image
        x = F.interpolate(y, size=(H_hr, W_hr), mode='bicubic', align_corners=False)
        x = torch.clamp(x, 0.0, 1.0)

        # FISTA variables
        z = x.clone()  # auxiliary variable
        t = 1.0  # momentum parameter

        for i in range(self.n_iter):
            x_old = x.clone()

            # --- Gradient of data-fidelity: ∇_x ½||Hx - y||² = H^T(Hx - y) ---
            Hz = self._forward(z)       # Apply H to current estimate
            residual = Hz - y           # (1, C, H_lr, W_lr)
            grad_data = self._adjoint(residual)  # H^T(Hx - y), shape (1, C, H_hr, W_hr)

            # --- Gradient of TV prior: ∇_x λ·TV(x) ---
            grad_tv = self._tv_gradient(z)  # (1, C, H_hr, W_hr)

            # --- Gradient descent step ---
            x = z - self.step_size * (grad_data + self.lambda_tv * grad_tv)
            x = torch.clamp(x, 0.0, 1.0)  # Project onto [0, 1]

            # --- FISTA momentum update ---
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            z = x + ((t - 1.0) / t_new) * (x - x_old)
            z = torch.clamp(z, 0.0, 1.0)
            t = t_new

        # Return (C, H_hr, W_hr)
        return x.squeeze(0)

    def _forward(self, x):
        """
        Apply the forward degradation operator H (blur + downsampling).

        Parameters
        ----------
        x : torch.Tensor
            High-resolution image, shape (B, C, H_hr, W_hr).

        Returns
        -------
        torch.Tensor
            Degraded image, shape (B, C, H_lr, W_lr).
        """
        B, C, H, W = x.shape

        # Apply Gaussian blur channel-wise
        blurred = torch.zeros_like(x)
        for c in range(C):
            blurred[:, c:c+1, :, :] = F.conv2d(
                x[:, c:c+1, :, :],
                self.kernel,
                padding=self.padding
            )

        # Downsample
        downsampled = F.interpolate(
            blurred,
            scale_factor=1.0 / self.sr_factor,
            mode='bilinear',
            align_corners=False
        )
        return downsampled

    def _adjoint(self, y):
        """
        Apply the adjoint operator H^T (upsampling + correlation with transposed kernel).

        The adjoint of bilinear downsampling is bilinear upsampling,
        and the adjoint of convolution with kernel k is correlation with k
        (i.e., convolution with flipped kernel).

        Parameters
        ----------
        y : torch.Tensor
            Low-resolution image, shape (B, C, H_lr, W_lr).

        Returns
        -------
        torch.Tensor
            Result in high-resolution space, shape (B, C, H_hr, W_hr).
        """
        B, C, H_lr, W_lr = y.shape
        H_hr = H_lr * self.sr_factor
        W_hr = W_lr * self.sr_factor

        # Adjoint of downsampling = upsampling
        upsampled = F.interpolate(
            y,
            size=(H_hr, W_hr),
            mode='bilinear',
            align_corners=False
        )

        # Adjoint of convolution = correlation (conv with flipped kernel)
        # For a symmetric Gaussian kernel, flip(k) = k, so adjoint = convolution
        result = torch.zeros_like(upsampled)
        for c in range(C):
            result[:, c:c+1, :, :] = F.conv2d(
                upsampled[:, c:c+1, :, :],
                self.kernel,  # Gaussian is symmetric, so adjoint = forward
                padding=self.padding
            )

        return result

    def _tv_gradient(self, x, epsilon=1e-6):
        """
        Compute the gradient of the anisotropic Total Variation.

        TV(x) = Σ_i,j (|x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|)

        The subgradient is approximated using a smoothed version:
        ∇TV ≈ -div(∇x / |∇x|)

        Parameters
        ----------
        x : torch.Tensor
            Image tensor, shape (B, C, H, W).
        epsilon : float
            Small constant for numerical stability.

        Returns
        -------
        torch.Tensor
            TV gradient, same shape as input.
        """
        # Compute spatial gradients
        # Horizontal gradient: diff along width (axis 3)
        dx = torch.zeros_like(x)
        dx[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]

        # Vertical gradient: diff along height (axis 2)
        dy = torch.zeros_like(x)
        dy[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]

        # Smoothed gradient magnitude
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + epsilon)

        # Normalized gradients
        nx = dx / grad_mag
        ny = dy / grad_mag

        # Divergence of normalized gradient field: -div(n)
        # div(n) = ∂nx/∂x + ∂ny/∂y
        div = torch.zeros_like(x)

        # ∂nx/∂x (horizontal divergence)
        div[:, :, :, 0] = nx[:, :, :, 0]
        div[:, :, :, 1:-1] = nx[:, :, :, 1:-1] - nx[:, :, :, :-2]
        div[:, :, :, -1] = -nx[:, :, :, -2]

        # ∂ny/∂y (vertical divergence)
        div[:, :, 0, :] += ny[:, :, 0, :]
        div[:, :, 1:-1, :] += ny[:, :, 1:-1, :] - ny[:, :, :-2, :]
        div[:, :, -1, :] += -ny[:, :, -2, :]

        return -div

    def _prepare_input(self, img):
        """Convert various input formats to a (1, C, H, W) tensor on device."""
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img)).float()
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = img.permute(2, 0, 1)

        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = img.permute(2, 0, 1)

        # Normalize to [0, 1] if needed
        img = default_normalizer.normalize(img)
        img = img.float().to(self.device)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img

    @staticmethod
    def _create_gaussian_kernel(kernel_size, sigma):
        """Create a 2D Gaussian kernel of shape (1, 1, k, k)."""
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
        return kernel_2d.unsqueeze(0).unsqueeze(0)


def solve_tv(degraded_img, sr_factor=2, noise_std=0.005,
             lambda_tv=0.01, n_iter=500, step_size=0.005, device='cpu'):
    """
    Convenience function: solve SR with TV regularization.

    Parameters
    ----------
    degraded_img : torch.Tensor or PIL.Image or numpy.ndarray
        Low-resolution degraded observation.
    sr_factor : int
        Super-resolution factor (2 or 4).
    noise_std : float
        Noise standard deviation (0.005 or 0.01).
    lambda_tv : float
        TV regularization weight.
    n_iter : int
        Number of iterations.
    step_size : float
        Gradient descent step size.
    device : str
        Device ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        Reconstructed image (C, 256, 256) in [0, 1].
    """
    solver = TVSuperResolution(
        sr_factor=sr_factor,
        noise_std=noise_std,
        lambda_tv=lambda_tv,
        n_iter=n_iter,
        step_size=step_size,
        device=device
    )
    return solver.solve(degraded_img)


if __name__ == "__main__":
    # Quick test: solve SR on one degraded image
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_dir = data_dir / "test"
    degraded_dir = data_dir / "test_sr2x_noise0005"

    # Load one degraded image
    degraded_files = sorted(degraded_dir.glob("*.png"))
    if degraded_files:
        img_path = degraded_files[0]
        print(f"Testing TV on: {img_path.name}")

        degraded = Image.open(img_path).convert('RGB')
        print(f"  Input size: {degraded.size}")

        result = solve_tv(degraded, sr_factor=2, noise_std=0.005,
                          lambda_tv=0.01, n_iter=200, step_size=0.005)
        print(f"  Output shape: {result.shape}")
        print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")

        # Save result
        out_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)
        out_img.save(data_dir / "tv_test_output.png")
        print(f"  Saved to: {data_dir / 'tv_test_output.png'}")

        # Compute metrics if available
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from metrics import compute_psnr, compute_ssim

            gt_path = test_dir / img_path.name
            if gt_path.exists():
                gt = Image.open(gt_path).convert('RGB')
                gt_tensor = torch.from_numpy(np.array(gt)).float().permute(2, 0, 1) / 255.0
                psnr = compute_psnr(result, gt_tensor)
                ssim = compute_ssim(result, gt_tensor)
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  SSIM: {ssim:.4f}")
        except ImportError:
            print("  (metrics.py not available, skipping evaluation)")
    else:
        print("No degraded images found. Run degradations.py first.")
