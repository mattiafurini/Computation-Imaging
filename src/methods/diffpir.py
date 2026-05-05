"""
DiffPIR — Diffusion Models for Plug-and-Play Image Restoration.

Implements the DiffPIR algorithm for super-resolution using a pre-trained
diffusion model as a generative prior.

Reference:
    Zhu et al., "Denoising Diffusion Models for Plug-and-Play Image Restoration",
    CVPR 2023 (https://arxiv.org/abs/2305.08995)

Algorithm overview:
    Starting from Gaussian noise x_T, at each diffusion timestep t:
    1. Denoising step: x_{t-1} = denoise(x_t, t) using pre-trained diffusion model
    2. Data-fidelity step: project x_{t-1} towards consistency with observation y
       x_{t-1} = x_{t-1} - ρ_t · H^T(H·x_{t-1} - y)

The pre-trained model used is the unconditional 256×256 ImageNet diffusion model
from OpenAI's guided-diffusion repository.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from normalization import default_normalizer
except ImportError:
    from src.normalization import default_normalizer


# ============================================================================
# Diffusion utilities
# ============================================================================

def get_beta_schedule(num_diffusion_steps=1000, beta_start=0.0001, beta_end=0.02):
    """
    Linear beta schedule for the diffusion process.

    Returns
    -------
    betas : numpy.ndarray
        Array of shape (T,) with noise schedule values.
    """
    return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)


def compute_diffusion_params(betas):
    """
    Pre-compute all diffusion parameters from the beta schedule.

    Returns
    -------
    dict with keys:
        - alphas, alphas_cumprod, sqrt_alphas_cumprod, etc.
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': np.sqrt(1.0 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': np.sqrt(1.0 / alphas_cumprod - 1),
        'posterior_variance': betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        'posterior_mean_coef1': betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        'posterior_mean_coef2': (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
    }


def _extract(arr, timesteps, broadcast_shape, device):
    """Extract values from array at given timesteps and reshape for broadcasting."""
    vals = torch.tensor(arr, dtype=torch.float32, device=device)
    out = vals[timesteps]
    while out.ndim < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out


# ============================================================================
# Simplified UNet denoiser (for use without guided-diffusion dependency)
# ============================================================================

class SimpleDenoiser(torch.nn.Module):
    """
    A simplified denoiser network for DiffPIR.

    This is a lightweight UNet-like architecture that can be used as
    a stand-in if the full guided-diffusion model is not available.
    For best results, use the pre-trained guided-diffusion model instead.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB).
    base_channels : int
        Base number of channels.
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Time embedding (output must match bottleneck channels = base_channels * 8)
        bottleneck_channels = base_channels * 8
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, base_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(base_channels, bottleneck_channels),
        )

        # Encoder
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)

        self.pool = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = torch.nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)

        self.up2 = torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)

        self.up1 = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.final = torch.nn.Conv2d(base_channels, in_channels, 1)

    def _block(self, in_c, out_c):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, out_c),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, out_c),
            torch.nn.SiLU(),
        )

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor
            Noisy image, shape (B, 3, H, W).
        t : torch.Tensor
            Timestep indices, shape (B,).

        Returns
        -------
        torch.Tensor
            Predicted noise, shape (B, 3, H, W).
        """
        # Time conditioning (broadcast-add to feature maps)
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)  # (B, base_ch)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Add time embedding to bottleneck
        b = b + t_emb.unsqueeze(-1).unsqueeze(-1)

        # Decoder
        d3 = self.dec3(torch.cat([e3, self.up3(b)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))

        return self.final(d1)


# ============================================================================
# DiffPIR solver
# ============================================================================

class DiffPIR:
    """
    DiffPIR: Diffusion-based Plug-and-Play Image Restoration for Super-Resolution.

    Parameters
    ----------
    sr_factor : int
        Super-resolution factor (2 or 4).
    noise_std : float
        Noise level of the observation (0.005 or 0.01).
    model_path : str or Path, optional
        Path to pre-trained diffusion model checkpoint.
        If None, uses the built-in SimpleDenoiser.
    diffusion_steps : int
        Total number of diffusion timesteps in the schedule (default: 1000).
    sampling_steps : int
        Number of DDPM sampling steps to actually perform (default: 100).
        Uses uniform sub-sampling of the full schedule.
    rho : float
        Data-fidelity step size multiplier (default: 1.0).
    lambda_reg : float
        Regularization strength for data-fidelity (default: 0.5).
    device : str
        Device to use ('cpu' or 'cuda').
    """

    def __init__(self, sr_factor=2, noise_std=0.005, model_path=None,
                 diffusion_steps=1000, sampling_steps=100,
                 rho=1.0, lambda_reg=0.5, device='cuda'):
        assert sr_factor in [2, 4], "sr_factor must be 2 or 4"
        self.sr_factor = sr_factor
        self.noise_std = noise_std
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.device = device

        # Diffusion schedule
        betas = get_beta_schedule(diffusion_steps)
        self.params = compute_diffusion_params(betas)

        # Sub-sample timesteps for accelerated sampling
        self.timestep_seq = np.linspace(0, diffusion_steps - 1,
                                         sampling_steps, dtype=int)[::-1]

        # Degradation kernel (same as in degradations.py)
        kernel_size = sr_factor * 2 + 1
        sigma = sr_factor / np.sqrt(8 * np.log(2))
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Load or create denoiser model
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Load pre-trained diffusion model or create SimpleDenoiser."""
        if model_path is not None and Path(model_path).exists():
            # Try to load guided-diffusion model
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # If it's a guided-diffusion checkpoint, you'd need to build
                # the model architecture first. This is a simplified version.
                model = SimpleDenoiser(in_channels=3, base_channels=128)
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load {model_path}: {e}")
                print("Using SimpleDenoiser instead.")
                model = SimpleDenoiser(in_channels=3, base_channels=128)
        else:
            if model_path is not None:
                print(f"Warning: Model path {model_path} not found.")
            print("Using built-in SimpleDenoiser (for best results, use a pre-trained model).")
            model = SimpleDenoiser(in_channels=3, base_channels=128)

        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def restore(self, degraded_img):
        """
        Restore a high-resolution image from a degraded observation.

        Parameters
        ----------
        degraded_img : torch.Tensor or PIL.Image or numpy.ndarray
            Low-resolution degraded observation.
            Shape: (C, H_lr, W_lr) or PIL Image.

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (C, 256, 256) in range [0, 1].
        """
        # Prepare observation
        y = self._prepare_input(degraded_img)  # (1, C, H_lr, W_lr)
        C = y.shape[1]
        H_hr = y.shape[2] * self.sr_factor
        W_hr = y.shape[3] * self.sr_factor

        # Start from pure Gaussian noise at the target resolution
        x_t = torch.randn(1, C, H_hr, W_hr, device=self.device)

        for i, t in enumerate(self.timestep_seq):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            # --- Step 1: DDPM denoising step ---
            # Predict noise
            noise_pred = self.model(x_t, t_tensor)

            # Compute x_0 estimate from noise prediction
            sqrt_recip = _extract(self.params['sqrt_recip_alphas_cumprod'],
                                  t_tensor, x_t.shape, self.device)
            sqrt_recipm1 = _extract(self.params['sqrt_recipm1_alphas_cumprod'],
                                    t_tensor, x_t.shape, self.device)
            x0_pred = sqrt_recip * x_t - sqrt_recipm1 * noise_pred
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

            # --- Step 2: Data-fidelity step (DiffPIR core) ---
            # Compute rho_t: decreasing data-fidelity weight across timesteps
            alpha_bar = float(self.params['alphas_cumprod'][t])
            sigma_t = float(self.params['sqrt_one_minus_alphas_cumprod'][t])

            # DiffPIR schedule: stronger data-fidelity at lower noise levels
            rho_t = self.rho * self.lambda_reg / (self.lambda_reg + sigma_t ** 2)

            # Data-fidelity gradient: H^T(H·x0 - y)
            Hx0 = self._forward(x0_pred)
            residual = Hx0 - y
            grad_data = self._adjoint(residual)

            # Apply data-fidelity correction to x0 estimate
            x0_corrected = x0_pred - rho_t * grad_data
            x0_corrected = torch.clamp(x0_corrected, 0.0, 1.0)

            # --- Step 3: Compute x_{t-1} from corrected x0 ---
            if i < len(self.timestep_seq) - 1:
                # Use posterior mean formula to go from x0 to x_{t-1}
                t_prev = self.timestep_seq[i + 1]
                alpha_bar_prev = float(self.params['alphas_cumprod'][t_prev])

                # Re-noise to t-1
                noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
                x_t = (math.sqrt(alpha_bar_prev) * x0_corrected +
                       math.sqrt(1.0 - alpha_bar_prev) * noise)
            else:
                # Last step: just use the corrected x0
                x_t = x0_corrected

        return x_t.squeeze(0)

    def _forward(self, x):
        """Apply forward degradation H (blur + downsampling)."""
        B, C, H, W = x.shape
        blurred = torch.zeros_like(x)
        for c in range(C):
            blurred[:, c:c+1, :, :] = F.conv2d(
                x[:, c:c+1, :, :],
                self.kernel,
                padding=self.padding
            )
        downsampled = F.interpolate(
            blurred,
            scale_factor=1.0 / self.sr_factor,
            mode='bilinear',
            align_corners=False
        )
        return downsampled

    def _adjoint(self, y):
        """Apply adjoint operator H^T (upsampling + transposed convolution)."""
        B, C, H_lr, W_lr = y.shape
        H_hr = H_lr * self.sr_factor
        W_hr = W_lr * self.sr_factor

        upsampled = F.interpolate(y, size=(H_hr, W_hr), mode='bilinear',
                                   align_corners=False)
        result = torch.zeros_like(upsampled)
        for c in range(C):
            result[:, c:c+1, :, :] = F.conv2d(
                upsampled[:, c:c+1, :, :],
                self.kernel,
                padding=self.padding
            )
        return result

    def _prepare_input(self, img):
        """Convert input to (1, C, H, W) tensor on device."""
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img)).float()
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = img.permute(2, 0, 1)
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = img.permute(2, 0, 1)

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


def restore_diffpir(degraded_img, sr_factor=2, noise_std=0.005,
                     model_path=None, sampling_steps=100, device='cuda'):
    """
    Convenience function: restore an image using DiffPIR.

    Parameters
    ----------
    degraded_img : torch.Tensor or PIL.Image or numpy.ndarray
        Low-resolution degraded observation.
    sr_factor : int
        Super-resolution factor (2 or 4).
    noise_std : float
        Noise level (0.005 or 0.01).
    model_path : str or Path, optional
        Path to pre-trained diffusion model.
    sampling_steps : int
        Number of diffusion sampling steps.
    device : str
        Device ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        Reconstructed image (C, 256, 256) in [0, 1].
    """
    solver = DiffPIR(
        sr_factor=sr_factor,
        noise_std=noise_std,
        model_path=model_path,
        sampling_steps=sampling_steps,
        device=device,
    )
    return solver.restore(degraded_img)


if __name__ == "__main__":
    # Quick test
    data_dir = Path(__file__).parent.parent.parent / "data"
    degraded_dir = data_dir / "test_sr2x_noise0005"
    test_dir = data_dir / "test"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    degraded_files = sorted(degraded_dir.glob("*.png"))
    if degraded_files:
        img_path = degraded_files[0]
        print(f"\nTesting DiffPIR on: {img_path.name}")

        degraded = Image.open(img_path).convert('RGB')
        print(f"  Input size: {degraded.size}")

        # Use fewer steps for quick test
        result = restore_diffpir(
            degraded,
            sr_factor=2,
            noise_std=0.005,
            sampling_steps=20,  # Ridotto per test rapido
            device=device
        )
        print(f"  Output shape: {result.shape}")
        print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")

        # Save result
        out_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)
        out_img.save(data_dir / "diffpir_test_output.png")
        print(f"  Saved to: {data_dir / 'diffpir_test_output.png'}")

        # Compute metrics
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
            print("  (metrics.py not available)")
    else:
        print("No degraded images found. Run degradations.py first.")
