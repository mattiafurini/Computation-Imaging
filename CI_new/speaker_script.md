# 🎤 Speaker Script (10-15 minutes total)

*This script is divided into two parts. **Speaker 1** will present the introduction, dataset, UNet, and final quantitative results. **Speaker 2** will present the mathematical approaches (TV), generative approaches (DiffPIR), and the visual trade-offs. Each slide takes about 45-60 seconds to explain.*

---

## Part 1: Speaker 1 (Approx. 5-6 minutes)

### Slide 1: Title
**Speaker 1:** 
"Good morning everyone. Today, my colleague and I are presenting our Computational Imaging project. We tackled the inverse problem of Super-Resolution to explore the perception-distortion trade-offs."

### Slide 2: The Inverse Problem
**Speaker 1:**
"We formulated the forward model mathematically as $y = Hx + e$. The operator $H$ represents a Gaussian blur followed by downsampling, and $e$ represents additive white Gaussian noise. Because this system is highly underdetermined, we lose the property of uniqueness."

### Slide 3: Project Overview
**Speaker 1:**
"To solve this ill-posed problem, we compare three fundamentally different approaches, each relying on a different type of prior knowledge: End-to-End learning with a UNet, a Variational approach using Total Variation, and a Generative approach using DiffPIR."

### Slide 4: Dataset & Setup
**Speaker 1:**
"To ensure a fair and deterministic comparison, we built a unified degradation pipeline. We used a subset of 4000 RGB images from the LSUN Church dataset. Importantly, we explicitly resized all images to 256x256 pixels to fit the networks' architectures. To comprehensively evaluate the models, we tested 4 distinct degradation scenarios by combining Super-Resolution factors of 2 and 4 with noise levels of 0.005 and 0.01."

### Slide 5: Method 1 - UNet Architecture
**Speaker 1:**
"The first approach we implemented is an End-to-End deep learning method. We built a custom Encoder-Decoder UNet architecture with skip connections to route high-frequency spatial details directly to the output."

### Slide 6: UNet Implementation Details
**Speaker 1:**
"We trained it from scratch on our Kaggle environment using an L1 Loss function over 50 epochs, optimizing with Adam. We specifically chose L1 loss because it enforces sparsity in the residual error better than MSE."

### Slide 7: UNet Theoretical Behavior
**Speaker 1:**
"From a theoretical standpoint, minimizing this loss forces the network to predict the conditional mean of all possible high-resolution patches. Because of this 'regression to the mean', the network is incredibly effective at removing noise, but it struggles to synthesize complex high-frequency details, resulting in over-smoothed textures."

---

## Part 2: Speaker 2 (Approx. 5-6 minutes)

### Slide 8: Method 2 - Variational (Total Variation)
**Speaker 2:**
"Thank you. The second approach we implemented is a classical Variational method. The objective function seeks to minimize the data-fitting error while applying Total Variation as a regularization prior to promote sparse spatial gradients."

### Slide 9: TV Implementation Details
**Speaker 2:**
"We solved this optimization problem via an iterative Proximal Gradient Descent scheme. We heuristically tuned the regularization weight lambda to 0.05. Furthermore, since standard TV is formulated for grayscale, we adapted the mathematical operators to process 3D RGB tensors by computing the divergence channel-by-channel."

### Slide 10: TV Theoretical Behavior
**Speaker 2:**
"The main theoretical assumption of Total Variation is that images are 'piecewise constant'. While excellent for piecewise-smooth signals, it is problematic for natural images. As a result, our TV reconstruction preserves sharp edges but creates a strong 'staircase' or 'watercolor' effect, destroying natural textures like the sky or trees."

### Slide 11: Method 3 - Generative (DiffPIR)
**Speaker 2:**
"Our third and final approach is DiffPIR, which uses an unconditional diffusion model as a generative prior. At each reverse diffusion step, it blends the generative denoising process with a gradient step that enforces data consistency."

### Slide 12: DiffPIR Implementation Details
**Speaker 2:**
"We leveraged a pre-trained model on 1000 diffusion steps, but we sampled efficiently using 50 DDIM steps to reduce inference time. The data consistency step is computed using the adjoint of our specific degradation operator $H$."

### Slide 13: DiffPIR Theoretical Behavior
**Speaker 2:**
"Unlike the UNet, DiffPIR does not predict an average image; it samples from the actual data distribution. This allows it to 'hallucinate' highly realistic, high-frequency details. However, balancing the generative prior with the data-consistency constraint can sometimes introduce minor generative noise or grid-like artifacts."

---

## Part 3: Wrapping Up (Speaker 1 & 2)

### Slide 14: Quantitative Results (Speaker 1)
**Speaker 1:**
"We evaluated our methods quantitatively across all 4 degradation scenarios. As you can see in the table, the UNet generally dominates the metrics, particularly in the easier scenarios, achieving the highest PSNR and SSIM. Total Variation performed worse mathematically, and DiffPIR scored the lowest. However, as the degradation becomes more severe (SR=4), you can see the gap in metrics shift, reflecting each method's structural limitations."

### Slide 15: Qualitative Comparison (Speaker 2)
**Speaker 2:**
"However, these metrics only tell half the story. As you can see in this visual comparison, DiffPIR generates incredibly sharp and realistic textures that the UNet completely smoothed out. This perfectly demonstrates the 'Perception-Distortion Trade-off' theorem we studied in class: DiffPIR sacrifices pixel-by-pixel mathematical accuracy (PSNR) to achieve much higher perceptual quality, while UNet does the exact opposite."

### Slide 16: Qualitative Breakdown under Severe Degradation (Speaker 2)
**Speaker 2:**
"Finally, we analyzed the visual breakdown of these methods under the most severe degradation: a Super-Resolution factor of 4 with a noise level of 0.01. The UNet underperforms structurally; it outputs a blurry image because it simply predicts a conservative average. Total Variation is mathematically consistent: it actively solves the inverse problem to recover edges, even if the watercolor effect is heavily amplified. DiffPIR struggles: the severe degradation causes a conflict between its generative prior and the data-fidelity gradient, leading to heavy hallucinatory noise and grid artifacts instead of clean textures."

### Slide 17: Conclusions (Speaker 1)
**Speaker 1:**
"To conclude, our comparative analysis confirms that there is no perfect method. End-to-end models are best for pure mathematical accuracy but lack robustness. Variational methods offer solid mathematical guarantees and robustness, but struggle with natural textures. Finally, Generative models provide state-of-the-art visual realism, but hallucinate under severe degradation."

### Slide 18: Future Developments (Speaker 1)
**Speaker 1:**
"For future developments, we look toward Algorithm Unfolding to combine variational guarantees with deep learning priors, or advanced diffusion solvers to better control the trade-off between hallucinations and physical data consistency. Our code is available on GitHub. Thank you for your attention!"
