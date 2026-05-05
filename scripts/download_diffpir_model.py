"""
Download script for the DiffPIR pre-trained denoiser model.

Model: google/ddpm-ema-church-256
  - Trained on: LSUN Church dataset (same domain as this project)
  - Architecture: DDPM UNet 256x256 with EMA weights
  - Size: ~400 MB
  - License: Apache 2.0

This script is called automatically by diffpir.py if the model is not found.
You can also run it manually:
    python scripts/download_diffpir_model.py
"""

import sys
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
MODEL_ID = "google/ddpm-ema-church-256"
MODEL_FILENAME = "ddpm_church_256_unet.pt"


def model_exists():
    """Check if the model has already been downloaded."""
    return (WEIGHTS_DIR / MODEL_FILENAME).exists()


def download_model(force=False):
    """
    Download the DDPM EMA Church 256x256 denoiser from HuggingFace.

    Parameters
    ----------
    force : bool
        If True, re-download even if the model already exists.

    Returns
    -------
    Path
        Path to the downloaded model file.
    """
    out_path = WEIGHTS_DIR / MODEL_FILENAME

    if out_path.exists() and not force:
        print(f"  [DiffPIR] Model already present: {out_path}")
        return out_path

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DiffPIR — Downloading pre-trained denoiser model")
    print(f"  Source : HuggingFace — {MODEL_ID}")
    print(f"  Domain : LSUN Church 256x256 (EMA weights)")
    print(f"  Size   : ~400 MB")
    print(f"  Dest   : {out_path}")
    print("=" * 60)

    try:
        import torch
        from diffusers import DDPMPipeline

        print("  Loading pipeline from HuggingFace...")
        pipe = DDPMPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        unet = pipe.unet
        unet.eval()

        print("  Saving UNet weights...")
        torch.save(unet.state_dict(), out_path)
        print(f"  OK — Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"  ERROR during download: {e}")
        print("  Try manually: pip install diffusers accelerate")
        raise


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = download_model(force=force)
    print(f"\nModel ready at: {path}")
