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
    * $x \in \mathbb{R}^N$: The unknown HR image (ground truth).
    * $y \in \mathbb{R}^M$: The measured LR observation (where $M \ll N$, severely underdetermined).
    * $H$: Degradation operator. A linear operator composed of a Gaussian Blur filter followed by spatial downsampling with stride $s$.
    * $e$: Stochastic noise. We model it as Additive White Gaussian Noise, though real sensors also exhibit Poisson (Shot) noise.
  * **Mathematical Nature:** Since $M < N$, the system is highly ill-posed. It admits infinite possible solutions and loses the property of uniqueness.

## Slide 3: Project Overview & Selected Methods
* **Title:** Project Overview & Methodologies
* **Bullets:**
  * To solve this ill-posed problem, we compare three fundamentally different approaches, each relying on a different type of *prior* knowledge:
  * **1. End-to-End Learning (UNet):** Learns the direct mapping from $y$ to $x$ using a supervised dataset.
  * **2. Variational Method (Total Variation):** Solves an optimization problem enforcing mathematical properties (gradient sparsity).
  * **3. Generative Model (DiffPIR):** Uses an unconditional diffusion model as a strong generative prior to hallucinate missing details.

## Slide 4: Dataset & Setup
* **Title:** Dataset & Uniform Degradation Pipeline
* **Bullets:**
  * **Dataset:** LSUN Church (Subset of 4000 RGB images).
  * **Preprocessing:** Images are explicitly resized to 256x256 pixels and normalized in the `[0, 1]` range (scaled to `[-1, 1]` for the diffusion model).
  * **Splits:** 3200 Train | 400 Validation | 400 Test.
  * **Degradation Grid:** To comprehensively evaluate the models, we test 4 degradation scenarios combining SR factors $\{2, 4\}$ and Noise levels $\{0.005, 0.01\}$.
* **Image:** Show a Ground Truth (GT) image next to its 4 Degraded versions.

## Slide 5: Method 1 - End-to-End Learning (UNet)
* **Title:** End-to-End Reconstruction: UNet
* **Bullets:**
  * **Category:** Supervised Image-to-Image mapping.
  * **Architecture Details:** Encoder-Decoder with Skip Connections (4 Downsampling/Upsampling blocks).
    * *Deep layers* capture the global semantic context at low resolution.
    * *Skip connections* route high-frequency spatial details directly to the decoder.
* **Image:** A simple block diagram of a UNet.

## Slide 6: UNet - Implementation & Training
* **Title:** UNet: Implementation Details
* **Bullets:**
  * **Loss Function:** L1 Loss. Chosen because it enforces sparsity in the residual error better than MSE.
  * **Optimizer:** Adam optimizer.
  * **Training Setup:** 50 Epochs on the training split (3200 images).

## Slide 7: UNet - Theoretical Behavior
* **Title:** UNet: "Regression to the Mean"
* **Bullets:**
  * Minimizing L1/L2 loss forces the network to predict the *conditional mean* of possible HR patches.
  * **Result:** Excellent noise removal, high PSNR, but over-smoothed textures.
* **Image:** Zoom-in on a UNet result showing smooth bricks/leaves.

## Slide 8: Method 2 - Variational (Total Variation)
* **Title:** Variational Reconstruction: Total Variation (TV)
* **Bullets:**
  * **Objective Function:** $\min_{x} \frac{1}{2}||y - Hx||_2^2 + \lambda TV(x)$
    * **Data Fitting:** Consistency with the measured LR observation.
    * **Prior:** $TV(x)$ promotes sparse spatial gradients.
* **Image:** A generic formula or illustration of spatial gradients.

## Slide 9: TV - Implementation Details
* **Title:** TV: Implementation Details
* **Bullets:**
  * **Solver:** Iterative scheme using FISTA / Proximal Gradient Descent.
  * **Hyperparameters:** $\lambda = 0.05$ (tuned on validation set to balance data-fitting and regularization). Maximum iterations set to 300.
  * **RGB Adaptation:** Channel-by-channel divergence to natively support 3D color tensors.

## Slide 10: TV - Theoretical Behavior
* **Title:** TV: The Staircase Effect
* **Bullets:**
  * TV mathematically assumes images are "piecewise constant".
  * **Tuning $\lambda$:**
    * $\lambda$ too small: Regularization vanishes, noise and artifacts dominate.
    * $\lambda$ too large: Triggers the **staircasing effect** (artificial blocky regions).
    * Our chosen $\lambda = 0.05$ represents the thermodynamic equilibrium between data consistency and gradient sparsity.
* **Image:** Zoom-in on a TV result showing the blocky/painted effect.

## Slide 11: Method 3 - Generative (DiffPIR)
* **Title:** Generative Reconstruction: DiffPIR
* **Bullets:**
  * **Prior:** Unconditional Diffusion Model (`ddpm-ema-church-256`).
  * **Mechanism:** Interleaves standard DDIM reverse diffusion steps with a Data-Fidelity gradient step.
  * **Data Consistency:** Uses the adjoint operator $H^T$ to guide the diffusion process toward the measured observation.

## Slide 12: DiffPIR - Implementation Details
* **Title:** DiffPIR: Implementation Details
* **Bullets:**
  * **Diffusion Steps:** The prior model is trained on 1000 steps.
  * **Sampling Strategy:** Sampled efficiently using 50 DDIM steps to reduce inference time while maintaining quality.
  * **Advantage:** Actively samples from the learned data distribution $p(x)$ to "hallucinate" plausible high-frequency details.

## Slide 13: DiffPIR - Theoretical Behavior
* **Title:** DiffPIR: Hallucinations & Generative Noise
* **Bullets:**
  * Does not predict the average; it predicts a *plausible* reality.
  * **Result:** Highly realistic textures (leaves, stones), but introduces minor generative noise/grid artifacts due to the data-consistency constraint.
* **Image:** Zoom-in on a DiffPIR result showing sharp textures but slight background noise.

## Slide 14: Quantitative Results & Metric Limits
* **Title:** Quantitative Evaluation & Metric Analysis
* **Text:** *Averaged over 50 test images due to inference time constraints of diffusion models.*
* **Table:**
  | Method | SR=2, n=0.005 | SR=2, n=0.01 | SR=4, n=0.005 | SR=4, n=0.01 |
  |---|---|---|---|---|
  | **UNet** | 27.20 / 0.8765 | 26.99 / 0.8609 | 22.21 / 0.6475 | 22.17 / 0.6381 |
  | **TV** | 24.28 / 0.7183 | 24.27 / 0.7179 | 22.26 / 0.6205 | 22.24 / 0.6196 |
  | **DiffPIR** | 24.00 / 0.7389 | 23.96 / 0.7331 | 20.85 / 0.5318 | 20.83 / 0.5274 |
* **Metric Discussion:**
  * **MSE/PSNR:** Calculates purely numeric pixel-by-pixel error. Has low correlation with human visual quality.
  * **SSIM (Structural Similarity):** More informative. Evaluates luminance, contrast, and structural geometry. UNet generally dominates here.

## Slide 15: Qualitative Comparison
* **Title:** Visual Comparison & Perception-Distortion Tradeoff
* **Image:** Insert the large 3x5 Grid Plot you generated on Kaggle! (`Original | Degraded | UNet | TV | DiffPIR`).
* **Bullets:** 
  * UNet: Cleanest (Best Distortion/PSNR).
  * DiffPIR: Sharpest (Best Perception, but worst PSNR).

## Slide 16: Qualitative Breakdown under Severe Degradation
* **Title:** Visual Breakdown (SR=4, Noise=0.01)
* **Image:** Insert the 3x5 Grid Plot from the most severe degradation test.
* **Bullets:** 
  * **UNet (End-to-End):** Underperforms structurally. Outputs a blurry image lacking high-frequency details.
  * **TV (Variational):** Mathematically consistent. Recovers edges effectively despite the larger blur, though the "watercolor" effect is heavily amplified.
  * **DiffPIR (Generative):** Hallucinates heavily. The severe degradation breaks the balance between data-fidelity and the diffusion prior, introducing high-frequency noise and grid artifacts instead of clean textures.

## Slide 17: Conclusions
* **Title:** Conclusions
* **Bullets:**
  * **End-to-End:** Best for pure mathematical accuracy and fast inference, but lacks robustness.
  * **Variational:** Reliable mathematical guarantees and highly robust, but unsuited for complex natural textures.
  * **Generative:** Unmatched perceptual quality, but computationally expensive, hallucinates artifacts under severe degradation.

## Slide 18: Future Developments
* **Title:** Future Developments
* **Bullets:**
  * **Algorithm Unfolding:** Unrolling TV gradient descent iterations inside network layers (combining variational guarantees with deep learning priors).
  * **Advanced Generative Models:** Further exploring DPS/DiffPIR to control the exact trade-off between network hallucinations and physical data consistency.
* **Text:** *Code available on GitHub. Thank you!*
