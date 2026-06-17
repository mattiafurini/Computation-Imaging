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
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%); /* A breath of very light green-blue */
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
  section.lead strong { color: #2563eb; } /* Names in bright electric blue */

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
  
  /* Box to highlight code or intervals like [0,1] */
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
    background-color: #f8fafc !important; /* Light gray header */
    color: #1e3a8a !important;            /* Dark blue text */
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
  /* Righe dei metodi con background colorati */
  tr:nth-child(1) td { 
    background-color: #f0fdf4 !important; /* UNet: Soft Mint Green */
    color: #15803d;
  }
  tr:nth-child(1) td:first-child strong { color: #15803d; }

  tr:nth-child(2) td { 
    background-color: #f0f9ff !important; /* TV: Soft Blue */
    color: #0369a1;
  }
  tr:nth-child(2) td:first-child strong { color: #0369a1; }

  tr:nth-child(3) td { 
    background-color: #f5f3ff !important; /* DiffPIR: Soft Purple */
    color: #6d28d9;
  }
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
  * $x \in \mathbb{R}^N$: Unknown HR image (ground truth).
  * $y \in \mathbb{R}^M$: Measured LR observation ($M \ll N$, severely underdetermined).
  * $H$: Degradation operator (Gaussian Blur + spatial downsampling with stride $s$).
  * $e$: Stochastic noise (`AWGN`).
* **Mathematical Nature:** Since $M < N$, the system is highly ill-posed. It admits infinite possible solutions and loses uniqueness.

---

## Project Overview & Methodologies

To solve this ill-posed problem, we compare three fundamentally different approaches, each relying on a different type of *prior* knowledge:

1. **End-to-End Learning (UNet):** Learns the direct mapping from $y$ to $x$ using a supervised dataset.
2. **Variational Method (Total Variation):** Solves an optimization problem enforcing mathematical properties (gradient sparsity).
3. **Generative Model (DiffPIR):** Uses an unconditional diffusion model as a strong generative prior to hallucinate missing details.

---

## Dataset & Uniform Degradation Pipeline

* **Dataset:** LSUN Church (Subset of 4000 RGB images).
* **Preprocessing:** Images resized to 256x256 pixels and normalized in `[0, 1]` (scaled to `[-1, 1]` for the diffusion model).
* **Splits:** 3200 Train | 400 Validation | 400 Test.
* **Degradation Grid:** 4 scenarios combining SR factors $\{2, 4\}$ and Noise levels $\{0.005, 0.01\}$.

![w:550](immagini%20presentazione/degradation_grid_image.png)

---

## Method 1 — End-to-End Learning (UNet)

* **Category:** Supervised Image-to-Image mapping.
* **Architecture Details:** Encoder-Decoder with Skip Connections (4 Downsampling/Upsampling blocks).
  * *Deep layers* capture the global semantic context at low resolution.
  * *Skip connections* route high-frequency spatial details directly to the decoder.

![w:450](path_to_unet_diagram.png)

---

## UNet: Implementation Details

* **Loss Function:** L1 Loss. Chosen because it enforces sparsity in the residual error better than MSE.
* **Optimizer:** Adam optimizer.
* **Training Setup:** 50 Epochs on the training split (3200 images).

---

## UNet: "Regression to the Mean"

* Minimizing L1/L2 loss forces the network to predict the *conditional mean* of possible HR patches.
* **Result:** Excellent noise removal, high PSNR, but over-smoothed textures.

![w:450](immagini%20presentazione/unet_zoom_smooth_bricks.png)

---

## Method 2 — Variational (Total Variation)

* **Objective Function:**
  $$\min_{x} \frac{1}{2}||y - Hx||_2^2 + \lambda TV(x)$$

* **Data Fitting:** Enforces consistency with the measured LR observation.
* **Prior:** $TV(x)$ promotes sparse spatial gradients.

---

## TV: Implementation Details

* **Solver:** Iterative scheme using FISTA / Proximal Gradient Descent.
* **Hyperparameters:** $\lambda = 0.05$ (tuned on validation set to balance data-fitting and regularization). Maximum iterations set to 300.
* **RGB Adaptation:** Channel-by-channel divergence to natively support 3D color tensors.

---

## TV: The Staircase Effect

* TV mathematically assumes images are "piecewise constant".
* **Tuning $\lambda$:**
  * $\lambda$ too small: Regularization vanishes, noise and artifacts dominate.
  * $\lambda$ too large: Triggers the **staircasing effect** (artificial blocky regions).
* Our $\lambda = 0.05$ represents the ideal balance between data consistency and gradient sparsity.

---

## Method 3 — Generative (DiffPIR)

* **Prior:** Unconditional Diffusion Model (`ddpm-ema-church-256`).
* **Mechanism:** Interleaves standard DDIM reverse diffusion steps with a Data-Fidelity gradient step.
* **Data Consistency:** Uses the adjoint operator $H^T$ to guide the diffusion process toward the measured observation $y$.

---

## DiffPIR: Implementation Details

* **Diffusion Steps:** The prior model is trained on 1000 steps.
* **Sampling Strategy:** Sampled efficiently using 50 DDIM steps to reduce inference time while maintaining quality.
* **Advantage:** Actively samples from the learned data distribution $p(x)$ to "hallucinate" plausible high-frequency details.

---

## DiffPIR: Hallucinations & Generative Noise

* Does not predict the statistical mean; predicts a *plausible* reality.
* **Result:** Highly realistic textures (leaves, stones), but introduces minor generative noise/grid artifacts due to the data-consistency constraint.

---

## Quantitative Evaluation & Metric Analysis

*Averaged over 50 test images due to inference time constraints.*

| Method | SR=2, n=0.005 | SR=2, n=0.01 | SR=4, n=0.005 | SR=4, n=0.01 |
| :--- | :---: | :---: | :---: | :---: |
| **UNet** | 27.20 / 0.8765 | 26.99 / 0.8609 | 22.21 / 0.6475 | 22.17 / 0.6381 |
| **TV** | 24.28 / 0.7183 | 24.27 / 0.7179 | 22.26 / 0.6205 | 22.24 / 0.6196 |
| **DiffPIR**| 24.00 / 0.7389 | 23.96 / 0.7331 | 20.85 / 0.5318 | 20.83 / 0.5274 |

* **MSE/PSNR:** Purely pixel-by-pixel error. Low correlation with human visual quality.
* **SSIM:** Evaluates luminance, contrast, and structural geometry. UNet dominates here.

---

## Visual Comparison & Perception-Distortion Tradeoff

* **UNet:** Cleanest overall (Best Distortion/PSNR).
* **DiffPIR:** Sharper in details (Best Perception, but PSNR is penalized).

![w:700](immagini%20presentazione/kaggl_3x5_grid_1.png)

---

## Visual Breakdown under Severe Degradation (SR=4, Noise=0.01)

* **UNet (End-to-End):** Structural collapse. Blurred output without high-frequency details.
* **TV (Variational):** Mathematically consistent. Recovers macro-edges, but the "watercolor" effect is amplified.
* **DiffPIR (Generative):** Heavy hallucinations. Severe degradation breaks the balance between data-fidelity and prior, introducing grid artifacts.

---

## Conclusions

* **End-to-End (UNet):** Ideal for pixel-by-pixel mathematical accuracy and real-time inference, but lacks structural robustness.
* **Variational (TV):** Solid mathematical guarantees and high robustness, but unsuitable for complex natural textures (staircasing).
* **Generative (DiffPIR):** Unparalleled perceptual quality and photorealism, but computationally heavy and prone to artifacts if the data is too degraded.

---

## Future Developments

* **Algorithm Unfolding:** Unroll the TV gradient descent iterations inside the layers of a neural network (combining variational guarantees and deep priors).
* **Advanced Generative Models:** Explore dynamic balancing in DPS/DiffPIR to control the exact trade-off between hallucination and physical consistency.

---

# Thank you!

*Code available on GitHub*

---

<!-- ================= BACKUP SLIDES ================= -->

# Backup Slides (Q&A)

---

## Backup: RGB Management in Total Variation

**Domanda frequente:** *La TV nasce per segnali 2D (grayscale). Come avete gestito i 3 canali RGB?*

* Nel nostro operatore `TVSuperResolution`, calcoliamo i gradienti spaziali separatamente lungo gli assi $x$ e $y$ usando differenze finite in PyTorch (`torch.roll`).
* Invece di accoppiare i canali (Vectorial TV), abbiamo applicato una **Channel-wise Anisotropic TV**:
  $$ TV(x) = \sum_{c \in \{R,G,B\}} \sum_{i,j} |x^c_{i+1,j} - x^c_{i,j}| + |x^c_{i,j+1} - x^c_{i,j}| $$
* Questo significa che i canali sono penalizzati in modo indipendente. Ha il vantaggio di essere velocissimo da calcolare, ma può talvolta causare lievi "sfasamenti di colore" sui bordi molto netti (color bleeding).

---

## Backup: DiffPIR Data Consistency Step

**Domanda frequente:** *Come fa il modello di diffusione a sapere quale immagine specifica deve ricostruire?*

* Il modello pre-addestrato fornisce solo il gradiente del prior $\nabla_x \log p(x)$.
* Per ancorare il risultato alla nostra misurazione $y$, DiffPIR intercala lo step di denoising con un **Data Consistency Step** basato sull'operatore lineare $H$ e il suo aggiunto $H^T$:
  $$ \hat{x}_0 = \arg\min_x ||y - Hx||^2_2 + \rho ||x - x_{pred}||^2_2 $$
* Calcolare questo step per il Gaussian Blur + Downsampling richiede una soluzione a forma chiusa (spesso nel dominio di Fourier via FFT) o iterazioni di gradiente. Questo garantisce che l'immagine "allucinata" $x$ collassi perfettamente nell'osservazione $y$ se degradata.

---

## Backup: The Degradation Operator ($H$)

**Domanda frequente:** *Come avete modellato l'operatore matematico H? Avete usato filtri spaziali o in frequenza?*

* Abbiamo costruito $H$ nel **dominio spaziale** usando `torch.nn.functional.conv2d`.
* Il blur è modellato con un kernel Gaussiano spaziale dipendente dal fattore di scala (es. kernel 5x5).
* Per evitare bordi neri o artefatti durante la convoluzione, abbiamo usato il padding `reflection`, che specchia i pixel ai bordi simulando una continuazione naturale dell'immagine.
* Il downsampling spaziale è un semplice campionamento regolare (stride $= s$).