---
marp: true
html: true
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
  .twocolumns {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 30px;
  }
  .twocolumns > div {
    flex: 1;
  }
  .twocolumns > div:first-child {
    flex: 0.9;
  }
  .twocolumns > div:last-child {
    flex: 1.1;
  }
  .twocolumns table {
    display: table;
    width: 100%;
    table-layout: auto;
  }
  
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
  h2 small {
    display: block;
    font-size: 0.6em;
    font-weight: 500;
    color: #64748b;
    margin-top: 5px;
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
Mohamed Samir Haffoudhi - mohamed.haffoudhi@studio.unibo.it - 0001242845
Mattia Furini - mattia.furini@studio.unibo.it - 0001233051

**GitHub Repo:** [Computational-Imaging-Group-I](https://github.com/Samirhff1/Computational-Imaging-Group-I-Project-Template.git)

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

1. **Supervised Post-Processing (UNet):** End-to-end mapping from corrupted images to Ground Truth.
2. **Total Variation (TV):** Variational optimization using the L1 norm on the image gradient.
3. **Diffusion Plug-and-Play (DiffPIR):** Alternates diffusion denoising with a variational step for data fidelity.

---

## Dataset & Pipeline

* **Dataset:** LSUN Church (4000 RGB images).
* **Preprocessing:** Resized to 256x256, normalized in `[0, 1]`.
* **Splits:** 3200 Train | 400 Val | 400 Test.
* **Degradation:** 4 scenarios (SR factors $\{2, 4\}$, Noise $\{0.005, 0.01\}$).

![w:900](immagini%20presentazione/degradation_grid_image.png)

---

## Method 1 — End-to-End (UNet)

* **Architecture:** Encoder-Decoder with Skip Connections.
* **Setup:** L1 Loss (enforces sparsity), Adam optimizer, 50 Epochs.
* **Mechanism:** Minimizing L1 predicts the *conditional mean* of HR patches.
* **Result:** Excellent noise removal, but over-smoothed textures.

![w:450](immagini%20presentazione/uNet-architecture.png)

---

## Method 2 — Variational (TV)

* **Objective Function:**
  $$\min_{x} \frac{1}{2}||y - Hx||_2^2 + \lambda TV(x)$$
* **Mechanism:** Data consistency with $y$ + prior $TV(x)$ promoting sparse spatial gradients.
* **Setup:** FISTA solver, $\lambda = 0.05$, Channel-by-channel divergence for RGB.
* **Result:** Preserves macro-edges but causes a "staircase effect" (artificial blocky regions).

---

## Method 3 — Generative (DiffPIR)

* **Prior:** Unconditional Diffusion Model (`ddpm-ema-church-256`).
* **Mechanism:** Interleaves DDIM denoising steps with Data-Fidelity guidance via $H^T$.
* **Setup:** 50 DDIM sampling steps.
* **Result:** Actively samples from $p(x)$ yielding highly realistic textures, but prone to hallucinations/artifacts.

---

## Evaluation: Mild Degradation <br> <small>(SR=2, Noise=0.005)</small>

<div class="twocolumns">

<div>

| Method | PSNR / SSIM |
| :--- | :---: |
| **UNet** | **26.94 / 0.8732** |
| **TV** | 24.28 / 0.7183 |
| **DiffPIR**| 24.00 / 0.7389 |

</div>

<div>

![w:650](immagini%20presentazione/kaggl_3x5_grid_sr2_n0.005.png)

</div>

</div>

---

## Evaluation: Moderate Degradation <br> <small>(SR=2, Noise=0.01)</small>

<div class="twocolumns">

<div>

| Method | PSNR / SSIM |
| :--- | :---: |
| **UNet** | **26.77 / 0.8617** |
| **TV** | 24.27 / 0.7179 |
| **DiffPIR**| 23.96 / 0.7331 |

</div>

<div>

![w:650](immagini%20presentazione/kaggl_3x5_grid_sr2_n0.01.png)

</div>

</div>

---

## Evaluation: High Degradation <br> <small>(SR=4, Noise=0.005)</small>

<div class="twocolumns">

<div>

| Method | PSNR / SSIM |
| :--- | :---: |
| **UNet** | **23.14 / 0.7003** |
| **TV** | 22.26 / 0.6205 |
| **DiffPIR**| 20.85 / 0.5318 |

</div>

<div>

![w:650](immagini%20presentazione/kaggl_3x5_grid_sr4_n0.005.png)

</div>

</div>

---

## Evaluation: Severe Degradation <br> <small>(SR=4, Noise=0.01)</small>

<div class="twocolumns">

<div>

| Method | PSNR / SSIM |
| :--- | :---: |
| **UNet** | **23.08 / 0.6913** |
| **TV** | 22.24 / 0.6196 |
| **DiffPIR**| 20.83 / 0.5274 |

</div>

<div>

![w:650](immagini%20presentazione/severe_degradation_breakdown.png)

</div>

</div>

---

## UNet: Robustness vs Over-smoothing

<div class="twocolumns">

<div>

* **Robustness:** Successfully handles multiple levels of degradation.
* **Over-smoothing:** Causes natural textures to look artificially flat compared to generative methods.
* **Generalization:** Struggles to reconstruct images outside its specific training distribution (OOD data).

</div>

<div>

![w:450](immagini%20presentazione/unet_zoom_smooth_bricks.png)

</div>

</div>

---

## TV Limitations: The "Watercolor" Effect

<div class="twocolumns">

<div>

* **"Watercolor" Effect:** Forces complex textures into unnatural, flat color blocks.
* **Staircase Artifacts:** Creates artificial blocky regions while preserving macro-edges, destroying photorealism.

</div>

<div>

![w:450](immagini%20presentazione/tv_zoom_watercolor.png)

</div>

</div>

---

## DiffPIR Limitations: Hallucinations

<div class="twocolumns">

<div>

* **Hallucinations:** Actively invents non-existent structures, especially when original information is completely lost (SR=4).
* **Grid Artifacts:** Unnatural repeating patterns emerge from the conflict between the generative prior and data consistency.

</div>

<div>

![w:450](immagini%20presentazione/diffpir_zoom_hallucinations.png)

</div>

</div>

---

## Conclusions

* **UNet:** Excellent metric performance across all degradations and fast inference, but outputs lack high-frequency photorealistic details.
* **TV:** Solid mathematical guarantees; unsuitable for natural textures.
* **DiffPIR:** Recreates highly realistic images, but produces hallucinations and artifacts under severe degradation.

---

# Thank you!

