# 🖥️ Google Slides Content (What to write on the slides)

*Note: Keep text on slides minimal. Use bullet points and large images.*

## Slide 1: Title
* **Title:** Super-Resolution on LSUN Church: A Comparative Analysis
* **Subtitle:** Computational Imaging - Group I
* **Text:** [Mattia's Name] & [Colleague's Name]

## Slide 2: The Inverse Problem
* **Title:** The Inverse Problem: Super-Resolution
* **Bullets:**
  * **Goal:** Reconstruct High-Resolution (HR) images from Low-Resolution (LR) degraded observations.
  * **Formulation:** $y = Hx + e$
  * **H:** Gaussian blur and downsampling (SR factor = 2).
  * **e:** Additive White Gaussian Noise ($\sigma = 0.005$).

## Slide 3: Dataset & Setup
* **Title:** Dataset & Uniform Degradation Pipeline
* **Bullets:**
  * **Dataset:** LSUN Church (Subset of 4000 RGB images).
  * **Format:** Natively 256x256 pixels.
  * **Splits:** 3200 Train | 400 Validation | 400 Test.
  * **Pipeline:** Unified `DegradationOperator` ensures all methods are evaluated on the *exact same* noisy inputs.
* **Image:** Show 1 Ground Truth (GT) image next to its Degraded (LR) version.

## Slide 4: Method 1 - End-to-End Learning (UNet)
* **Title:** End-to-End Reconstruction: UNet
* **Bullets:**
  * **Architecture:** Encoder-Decoder with skip connections.
  * **Training:** L1 Loss, Adam Optimizer, 50 Epochs.
  * **Hardware:** Kaggle Dual-GPU (`nn.DataParallel`).
* **Image:** A simple block diagram of a UNet (or just a nice icon representing deep learning).

## Slide 5: UNet - Theoretical Behavior
* **Title:** UNet: "Regression to the Mean"
* **Bullets:**
  * Minimizing L1/L2 loss forces the network to predict the *conditional mean* of possible HR patches.
  * **Result:** Excellent noise removal, high PSNR, but over-smoothed textures.
* **Image:** Zoom-in on a UNet result showing smooth bricks/leaves.

## Slide 6: Method 2 - Variational (Total Variation)
* **Title:** Variational Reconstruction: Total Variation (TV)
* **Bullets:**
  * **Solver:** FISTA algorithm.
  * **Prior:** Penalizes the image gradient to preserve sharp edges while smoothing flat regions.
  * **RGB Adaptation:** Channel-by-channel divergence to natively support 3D color tensors.
  * **Parameter:** $\lambda = 0.05$ (Heuristic balance).

## Slide 7: TV - Theoretical Behavior
* **Title:** TV: The Staircase Effect
* **Bullets:**
  * TV assumes images are "piecewise constant".
  * **Result:** Excellent on artificial structures/medical scans, but struggles on natural textures.
  * Creates an artificial "watercolor" or plastic effect on trees and skies.
* **Image:** Zoom-in on a TV result showing the blocky/painted effect.

## Slide 8: Method 3 - Generative (DiffPIR)
* **Title:** Generative Reconstruction: DiffPIR
* **Bullets:**
  * **Prior:** Unconditional Diffusion Model (`ddpm-ema-church-256`).
  * **Mechanism:** Blends reverse diffusion steps with a Data-Fidelity gradient step.
  * **Advantage:** Actively samples from the data distribution to "hallucinate" missing high-frequency details.

## Slide 9: DiffPIR - Theoretical Behavior
* **Title:** DiffPIR: Hallucinations & Generative Noise
* **Bullets:**
  * Does not predict the average; it predicts a *plausible* reality.
  * **Result:** Highly realistic textures (leaves, stones), but introduces minor generative noise/grid artifacts due to the data-consistency constraint.
* **Image:** Zoom-in on a DiffPIR result showing sharp textures but slight background noise.

## Slide 10: Quantitative Results
* **Title:** Quantitative Evaluation (Test Set)
* **Text:** *Averaged over 50 test images due to inference time constraints of diffusion models.*
* **Table:**
  * **UNet:** PSNR: 25.90 dB | SSIM: 0.8479
  * **TV:** PSNR: 24.66 dB | SSIM: 0.7204
  * **DiffPIR:** PSNR: 23.38 dB | SSIM: 0.6941

## Slide 11: Qualitative Comparison
* **Title:** Visual Comparison & Perception-Distortion Tradeoff
* **Image:** Insert the large 3x5 Grid Plot you generated on Kaggle! (`Original | Degraded | UNet | TV | DiffPIR`).
* **Bullets:** 
  * UNet: Cleanest (Best Distortion/PSNR).
  * DiffPIR: Sharpest (Best Perception, but worst PSNR).

## Slide 12: Robustness to Severe Degradations
* **Title:** Robustness Test: Out-of-Distribution Degradations (SR=4)
* **Image:** Insert the 3x5 Grid Plot from the Severe Degradation test.
* **Bullets:** 
  * **UNet (End-to-End):** Completely collapses. Outputs a blurry image because it cannot generalize to unseen degradation scales.
  * **TV (Variational):** Mathematically robust. Recovers edges effectively despite the larger blur, though the "watercolor" effect is amplified.
  * **DiffPIR (Generative):** Hallucinates heavily. The severe degradation breaks the balance between data-fidelity and the diffusion prior, introducing high-frequency noise and grid artifacts.

## Slide 13: Conclusions
* **Title:** Conclusions
* **Bullets:**
  * **End-to-End:** Best for pure mathematical accuracy and fast inference, but lacks robustness.
  * **Variational:** Reliable mathematical guarantees and highly robust, but unsuited for complex natural textures.
  * **Generative:** Unmatched perceptual quality, but computationally expensive, hallucinates artifacts under severe degradation, and sacrifices pixel-wise fidelity.
* **Text:** *Code available on GitHub. Thank you!*
