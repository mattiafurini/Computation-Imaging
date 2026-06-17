---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #334155
style: |
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
  
  section {
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 60px;
    background: #ffffff;
  }
  
  /* --- TOTAL LIGHT COVER --- */
  section.lead {
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%);
    color: #1e293b;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.lead h1 { 
    color: #1e3a8a; 
    font-size: 2.6em;
    font-weight: 800; 
    letter-spacing: -1px;
    border-bottom: none;
    margin-bottom: 10px;
  }
  section.lead h2 { 
    color: #475569; 
    font-size: 1.3em;
    font-weight: 500;
    border: none;
    padding: 0;
    margin-top: 0;
  }
  section.lead strong { color: #2563eb; }

  /* --- INTERNAL SLIDES --- */
  h2 { 
    color: #1e3a8a; 
    font-size: 1.9em;
    font-weight: 700;
    letter-spacing: -0.5px;
    border-bottom: 3px solid #3b82f6; 
    padding-bottom: 10px;
    margin-bottom: 25px;
  }
  
  ul { margin-top: 10px; }
  li { margin-bottom: 8px; line-height: 1.5; color: #334155; font-size: 0.95em; }
  li strong { color: #0f172a; font-weight: 700; }
  
  code {
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 0.85em;
  }
  
  /* --- LIGHT TABLE --- */
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin: 20px 0;
    border: 1px solid #e2e8f0;
  }
  th {
    background-color: #f8fafc !important; 
    color: #1e3a8a !important;            
    font-weight: 700;
    padding: 12px;
    font-size: 0.9em;
    border-bottom: 2px solid #cbd5e1 !important;
  }
  td {
    padding: 12px;
    background-color: #ffffff;
    border-bottom: 1px solid #f1f5f9 !important;
    font-size: 0.9em;
    color: #334155;
  }
  tr:nth-child(1) td { background-color: #f0fdf4 !important; color: #15803d; }
  tr:nth-child(1) td:first-child strong { color: #15803d; }
  tr:nth-child(2) td { background-color: #f0f9ff !important; color: #0369a1; }
  tr:nth-child(2) td:first-child strong { color: #0369a1; }
  tr:nth-child(3) td { background-color: #f5f3ff !important; color: #6d28d9; }
  tr:nth-child(3) td:first-child strong { color: #6d28d9; }
  
  footer { font-size: 0.6em; color: #94a3b8; font-weight: 500; }
  
  img {
    display: block;
    margin: 0 auto 30px auto;
  }
---

# Super-Resolution on LSUN Church
## A Comparative Analysis

**Computational Imaging - Group I**
Mattia Furini - 00000000
Samir - 00000

---

## The Inverse Problem: Super-Resolution

* **Goal:** Reconstruct High-Resolution (HR) images from Low-Resolution (LR) degraded observations.
* **Formalization:** $y = Hx + e$
  * $x \in \mathbb{R}^N$: HR image (ground truth).
  * $y \in \mathbb{R}^M$: LR observation ($M \ll N$).
  * $H$: Degradation (Gaussian Blur + downsampling).
  * $e$: Stochastic noise (`AWGN`).
* **Challenge:** System is highly ill-posed; infinite solutions exist.

---

## Project Overview

We compare three distinct approaches using different *priors*:

1. **End-to-End Learning (UNet):** Supervised mapping from $y$ to $x$.
2. **Variational Method (Total Variation):** Optimization enforcing gradient sparsity.
3. **Generative Model (DiffPIR):** Diffusion model as a generative prior for hallucinating details.

---

## Dataset & Pipeline

* **Dataset:** LSUN Church (4000 RGB images).
* **Preprocessing:** Resized to 256x256, normalized in `[0, 1]`.
* **Splits:** 3200 Train | 400 Val | 400 Test.
* **Degradation:** 4 scenarios (SR factors $\{2, 4\}$, Noise $\{0.005, 0.01\}$).

![w:550](immagini%20presentazione/degradation_grid_image.png)

---

## Method 1 — End-to-End (UNet)

* **Architecture:** Encoder-Decoder with Skip Connections.
  * *Deep layers:* Global semantic context.
  * *Skip connections:* High-frequency spatial details.

![w:450](path_to_unet_diagram.png)

---

## UNet: Details & Behavior

* **Setup:** L1 Loss (enforces sparsity), Adam optimizer, 50 Epochs.
* **"Regression to the Mean":** Minimizing L1 predicts the *conditional mean* of HR patches.
* **Result:** Excellent noise removal, but over-smoothed textures.

![w:450](immagini%20presentazione/unet_zoom_smooth_bricks.png)

---

## Method 2 — Variational (TV)

* **Objective Function:**
  $$\min_{x} \frac{1}{2}||y - Hx||_2^2 + \lambda TV(x)$$

* **Data Fitting:** Consistency with LR observation $y$.
* **Prior:** $TV(x)$ promotes sparse spatial gradients (piecewise constant).

---

## TV: Details & Behavior

* **Setup:** FISTA solver, $\lambda = 0.05$, Channel-by-channel divergence for RGB.
* **The Staircase Effect:** 
  * Assumes "piecewise constant" images.
  * Causes artificial blocky regions while preserving macro-edges.

---

## Method 3 — Generative (DiffPIR)

* **Prior:** Unconditional Diffusion Model (`ddpm-ema-church-256`).
* **Mechanism:** Interleaves DDIM denoising steps with a Data-Fidelity gradient step.
* **Data Consistency:** Uses adjoint $H^T$ to guide generation toward $y$.

---

## DiffPIR: Details & Behavior

* **Setup:** 50 DDIM sampling steps.
* **Hallucinations:** Actively samples from $p(x)$ rather than averaging.
* **Result:** Highly realistic textures, but prone to generative noise/grid artifacts.

---

## Quantitative Evaluation

*Average over 50 test images.*

| Method | SR=2, n=0.005 | SR=2, n=0.01 | SR=4, n=0.005 | SR=4, n=0.01 |
| :--- | :---: | :---: | :---: | :---: |
| **UNet** | 27.20 / 0.8765 | 26.99 / 0.8609 | 22.21 / 0.6475 | 22.17 / 0.6381 |
| **TV** | 24.28 / 0.7183 | 24.27 / 0.7179 | 22.26 / 0.6205 | 22.24 / 0.6196 |
| **DiffPIR**| 24.00 / 0.7389 | 23.96 / 0.7331 | 20.85 / 0.5318 | 20.83 / 0.5274 |

* **UNet** dominates MSE/PSNR and SSIM, despite lower visual realism.

---

## Perception-Distortion Tradeoff

* **UNet:** Cleanest (Best PSNR), over-smoothed.
* **DiffPIR:** Sharpest details (Best Perception), lower PSNR.

![w:700](immagini%20presentazione/kaggl_3x5_grid_1.png)

---

## Visual Breakdown (Severe Degradation)
*(SR=4, Noise=0.01)*

* **UNet:** Structural collapse, highly blurred.
* **TV:** Mathematically consistent edges, amplified "watercolor" effect.
* **DiffPIR:** Heavy hallucinations and grid artifacts due to prior/data-fidelity conflict.

---

## Conclusions

* **UNet:** Ideal for mathematical accuracy and speed; lacks structural robustness.
* **TV:** Solid mathematical guarantees; unsuitable for natural textures.
* **DiffPIR:** State-of-the-art photorealism; heavy compute and artifact-prone under severe degradation.

---

## Future Developments

* **Algorithm Unfolding:** Combine TV gradient steps with neural networks.
* **Advanced Generative Models:** Dynamic trade-off control between hallucination and physical consistency.

---

# Thank you!

*Code available on GitHub*

---

<!-- ================= BACKUP SLIDES ================= -->

# Backup Slides (Q&A)

---

## Backup: RGB in Total Variation

* TV computed separately along $x$ and $y$ using finite differences.
* **Channel-wise Anisotropic TV**: Independent penalty for R, G, B.
* Fast, but may cause slight color bleeding on sharp edges.

---

## Backup: DiffPIR Data Consistency

* Pre-trained model provides prior gradient $\nabla_x \log p(x)$.
* **Data Consistency Step:** Interleaved to anchor to $y$:
  $$ \hat{x}_0 = \arg\min_x ||y - Hx||^2_2 + \rho ||x - x_{pred}||^2_2 $$
* Ensures hallucinated $x$ collapses exactly into $y$ when degraded.

---

## Backup: The Degradation Operator ($H$)

* Built in spatial domain via `conv2d`.
* Gaussian blur kernel dependent on scale factor.
* **Padding `reflection`:** Avoids black borders.
* Downsampling via regular sampling (stride = $s$).