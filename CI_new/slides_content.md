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
  * **Formalization:** $y = Hx + e$
    * $x \in \mathbb{R}^N$: The unknown HR image (ground truth, latent vector).
    * $y \in \mathbb{R}^M$: The measured LR observation (where $M \ll N$, severely underdetermined).
    * $H$: Degradation operator (Gaussian blur + downsampling).
    * $e$: Stochastic noise. We model it as Additive White Gaussian Noise, though real sensors also exhibit Poisson (Shot) noise.
  * **Mathematical Nature:** Since $M < N$, the system is highly ill-posed. It admits infinite possible solutions and loses the property of uniqueness.

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
  * **Category:** Supervised End-to-End Image-to-Image mapping (Modulo 2).
  * **Architecture:** Encoder-Decoder with Skip Connections.
    * *Deep layers* capture the global semantic context at low resolution.
    * *Skip connections* route high-frequency spatial details directly to the decoder.
  * **Training:** L1 Loss, Adam Optimizer, 50 Epochs. L1 enforces sparsity in the residual error.
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
  * **Objective Function:** $\min_{x} \frac{1}{2}||y - Hx||_2^2 + \lambda TV(x)$
    * **Data Fitting:** $\frac{1}{2}||y - Hx||_2^2$ (Consistency with measurements).
    * **Prior:** $TV(x)$ (A priori info promoting sparse spatial gradients).
  * **Solver:** FISTA algorithm.
  * **RGB Adaptation:** Channel-by-channel divergence to natively support 3D color tensors.

## Slide 7: TV - Theoretical Behavior
* **Title:** TV: The Staircase Effect
* **Bullets:**
  * TV mathematically assumes images are "piecewise constant".
  * **Tuning $\lambda$:**
    * $\lambda$ too small: Regularization vanishes, noise and artifacts dominate.
    * $\lambda$ too large: Triggers the **staircasing effect** (artificial blocky regions).
    * Our chosen $\lambda = 0.05$ represents the thermodynamic equilibrium between data consistency and gradient sparsity.
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

## Slide 10: Quantitative Results & Metric Limits
* **Title:** Quantitative Evaluation & Metric Analysis
* **Text:** *Averaged over 50 test images due to inference time constraints of diffusion models.*
* **Table:**
  * **UNet:** PSNR: 25.90 dB | SSIM: 0.8479
  * **TV:** PSNR: 24.66 dB | SSIM: 0.7204
  * **DiffPIR:** PSNR: 23.38 dB | SSIM: 0.6941
* **Metric Discussion:**
  * **MSE/PSNR:** Calculates purely numeric pixel-by-pixel error. Has low correlation with human visual quality.
  * **SSIM (Structural Similarity):** More informative. Evaluates luminance, contrast, and structural geometry. UNet dominates here because it reconstructs structural coherence.

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

## Slide 13: Conclusions & Future Developments
* **Title:** Conclusions & Future Developments
* **Conclusions:**
  * **End-to-End:** Best for pure mathematical accuracy and fast inference, but lacks robustness.
  * **Variational:** Reliable mathematical guarantees and highly robust, but unsuited for complex natural textures.
  * **Generative:** Unmatched perceptual quality, but computationally expensive, hallucinates artifacts under severe degradation.
* **Future Developments:**
  * **Algorithm Unfolding:** Unrolling TV gradient descent iterations inside network layers (combining variational guarantees with deep learning priors).
  * **Advanced Generative Models:** Further exploring DPS/DiffPIR to control the exact trade-off between network hallucinations and physical data consistency.
* **Text:** *Code available on GitHub. Thank you!*
