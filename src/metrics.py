"""
Evaluation Metrics for Super-Resolution.

Implements PSNR and SSIM for quantitative comparison of reconstruction methods.
All metrics expect images in [0, 1] range as torch.Tensor or numpy.ndarray.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr
from pathlib import Path
from PIL import Image


def _to_numpy(img):
    """
    Convert image to numpy array in (H, W, C) format, range [0, 1].

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray or PIL.Image
        Input image.

    Returns
    -------
    numpy.ndarray
        Image array of shape (H, W, C) in [0, 1].
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0)
        img = img.numpy()
    elif isinstance(img, Image.Image):
        img = np.array(img).astype(np.float64) / 255.0

    img = np.asarray(img, dtype=np.float64)

    # Se ancora in range [0, 255], normalizza
    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def compute_psnr(pred, target):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between prediction and target.

    PSNR = 10 * log10(MAX² / MSE)

    Parameters
    ----------
    pred : torch.Tensor or numpy.ndarray
        Predicted (reconstructed) image, range [0, 1].
    target : torch.Tensor or numpy.ndarray
        Ground truth image, range [0, 1].

    Returns
    -------
    float
        PSNR value in dB. Higher is better.
    """
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    return float(_psnr(target_np, pred_np, data_range=1.0))


def compute_ssim(pred, target):
    """
    Compute Structural Similarity Index (SSIM) between prediction and target.

    Parameters
    ----------
    pred : torch.Tensor or numpy.ndarray
        Predicted (reconstructed) image, range [0, 1].
    target : torch.Tensor or numpy.ndarray
        Ground truth image, range [0, 1].

    Returns
    -------
    float
        SSIM value in [0, 1]. Higher is better.
    """
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    return float(_ssim(target_np, pred_np, data_range=1.0, channel_axis=2))


def evaluate_batch(pred_dir, target_dir, max_images=None):
    """
    Compute PSNR and SSIM over a directory of images.

    Images are matched by filename. Both directories must contain
    PNG or JPG files with identical names.

    Parameters
    ----------
    pred_dir : str or Path
        Directory containing predicted/reconstructed images.
    target_dir : str or Path
        Directory containing ground truth images.
    max_images : int, optional
        Maximum number of images to evaluate. If None, evaluate all.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'psnr_mean', 'psnr_std': mean and std of PSNR (dB)
        - 'ssim_mean', 'ssim_std': mean and std of SSIM
        - 'psnr_values': list of per-image PSNR
        - 'ssim_values': list of per-image SSIM
        - 'filenames': list of evaluated filenames
    """
    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)

    pred_files = sorted(pred_dir.glob('*.png')) + sorted(pred_dir.glob('*.jpg'))
    if max_images is not None:
        pred_files = pred_files[:max_images]

    psnr_values = []
    ssim_values = []
    filenames = []

    for pred_path in pred_files:
        target_path = target_dir / pred_path.name
        if not target_path.exists():
            continue

        pred_img = np.array(Image.open(pred_path).convert('RGB')).astype(np.float64) / 255.0
        target_img = np.array(Image.open(target_path).convert('RGB')).astype(np.float64) / 255.0

        # Le dimensioni devono corrispondere
        if pred_img.shape != target_img.shape:
            continue

        psnr_val = float(_psnr(target_img, pred_img, data_range=1.0))
        ssim_val = float(_ssim(target_img, pred_img, data_range=1.0, channel_axis=2))

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        filenames.append(pred_path.name)

    if not psnr_values:
        return {
            'psnr_mean': 0.0, 'psnr_std': 0.0,
            'ssim_mean': 0.0, 'ssim_std': 0.0,
            'psnr_values': [], 'ssim_values': [],
            'filenames': [],
        }

    return {
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'filenames': filenames,
    }
