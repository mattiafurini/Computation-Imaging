# 🎤 Speaker Script (10-15 minutes)

*Speaker 1: Introduction, dataset, UNet, quantitative results. Speaker 2: TV, DiffPIR, qualitative analysis. ~45-60s per slide.*

---

## Part 1: Speaker 1 (5-6 min)

### Slide 1: Title
"Good morning. Today we present our Computational Imaging project on Super-Resolution, exploring the perception-distortion trade-off."

### Slide 2: The Inverse Problem
"Our model is $y = Hx + e$, where $H$ is blur plus downsampling, and $e$ is noise. This system is heavily underdetermined, losing uniqueness."

### Slide 3: Project Overview
"We compare three approaches using different priors: a UNet (End-to-End), Total Variation (Variational), and DiffPIR (Generative)."

### Slide 4: Dataset & Setup
"We created a unified degradation pipeline on 4000 resized LSUN Church images. We tested 4 degradation scenarios combining Super-Resolution factors of 2 and 4 with noise levels of 0.005 and 0.01."

### Slide 5: Method 1 - UNet Architecture
"Our first approach is an End-to-End UNet with skip connections to directly route high-frequency details to the output."

### Slide 6: UNet Implementation Details
"Trained from scratch using Adam and L1 Loss, chosen specifically to better enforce sparsity in the residual error."

### Slide 7: UNet Theoretical Behavior
"Minimizing L1 predicts the conditional mean of possible high-resolution patches. This 'regression to the mean' effectively removes noise but yields over-smoothed textures."

---

## Part 2: Speaker 2 (5-6 min)

### Slide 8: Method 2 - Total Variation
"The second method is a Variational approach minimizing data-fitting error, using Total Variation to promote sparse spatial gradients."

### Slide 9: TV Implementation Details
"Solved via Proximal Gradient Descent. We tuned lambda to 0.05 and adapted the divergence operator channel-by-channel for RGB tensors."

### Slide 10: TV Theoretical Behavior
"TV assumes images are 'piecewise constant'. This preserves edges but creates a 'watercolor' or 'staircase' effect, destroying natural textures."

### Slide 11: Method 3 - DiffPIR
"Our final approach, DiffPIR, uses an unconditional diffusion model prior, blending denoising with a data consistency gradient step."

### Slide 12: DiffPIR Implementation Details
"We used 50 DDIM sampling steps for efficiency. The consistency step uses the adjoint operator $H^T$ to match our degradation."

### Slide 13: DiffPIR Theoretical Behavior
"Instead of an average, DiffPIR samples from the data distribution to 'hallucinate' realistic details. However, balancing this with data consistency can introduce grid artifacts."

---

## Part 3: Wrapping Up

### Slide 14: Quantitative Results (Speaker 1)
"Quantitatively, UNet dominates PSNR and SSIM across scenarios. TV and DiffPIR score lower. Under severe degradation (SR=4), performance drops significantly across all models."

### Slide 15: Qualitative Comparison (Speaker 2)
"Metrics aren't everything. DiffPIR generates sharp, realistic textures that UNet smooths out. This illustrates the 'Perception-Distortion Trade-off': DiffPIR trades PSNR for perceptual quality."

### Slide 16: Qualitative Breakdown (Speaker 2)
"Under severe degradation (SR=4, Noise=0.01): UNet blurs heavily. TV recovers edges but amplifies the watercolor effect. DiffPIR's prior clashes with data fidelity, causing heavy hallucinatory noise."

### Slide 17: Conclusions (Speaker 1)
"In conclusion: UNet offers accuracy but lacks robustness; TV gives mathematical guarantees but fails on textures; DiffPIR provides photorealism but hallucinates under severe degradation."

### Slide 18: Future Developments (Speaker 1)
"Future work could explore Algorithm Unfolding to combine variational and deep learning methods, or advanced diffusion solvers. Thank you!"
