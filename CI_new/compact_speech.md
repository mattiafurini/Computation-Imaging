# Compact Presentation Speech 🎤

*Here is a simple, direct speech in English. It avoids complex jargon and focuses on clearly explaining what you did in each slide.*

---

**Slide 1: Title**
Hello everyone. Today we will present our comparative analysis on Super-Resolution for Computational Imaging, focusing on the LSUN Church dataset.

**Slide 2: The Inverse Problem**
Our goal is to reconstruct high-resolution images from degraded, low-resolution ones. This is mathematically formulated as an inverse problem where we apply blur and add noise. It's a highly ill-posed challenge because infinite possible high-resolution images could result in the same blurry input.

**Slide 3: Project Overview**
To tackle this, we compared three distinct approaches: a Supervised Post-Processing neural network (UNet), a mathematical Variational method applying the L1 norm to the gradient (Total Variation), and a Diffusion Plug-and-Play generative method (DiffPIR).

**Slide 4: Dataset & Pipeline**
We used the LSUN Church dataset, testing our models on four different degradation scenarios. By varying the super-resolution scale factor and the noise level, we evaluated how each method handles different degrees of difficulty.

**Slide 5: Method 1 - UNet**
Our first method is the UNet. We trained it end-to-end using the L1 Loss. By minimizing this loss, the network learns to predict the conditional mean of the data. This gives excellent noise removal, but tends to produce over-smoothed textures.

**Slide 6: Method 2 - Total Variation (TV)**
The second method relies purely on mathematical optimization. Total Variation combines data consistency with a prior that favors sparse spatial gradients. It preserves the main macro-edges well, but can cause an artificial blocky, staircase effect.

**Slide 7: Method 3 - Generative (DiffPIR)**
Our third method is DiffPIR. It leverages a pre-trained Diffusion Model as a generative prior. By interleaving denoising steps with data-fidelity guidance, it actively samples from a distribution of realistic churches, allowing it to hallucinate incredible photorealistic details.

**Slide 8: Evaluation (Mild Degradation)**
Looking at the results for mild degradation, UNet quantitatively dominates with the highest PSNR and SSIM. However, visually, DiffPIR produces a much sharper and natural-looking image, highlighting the classic trade-off between mathematical distortion and human perception.

**Slide 9: Evaluation (Moderate Degradation)**
When we increase the noise slightly, the overall metrics drop across the board, but the relative behaviors remain consistent. UNet still leads in metrics, while DiffPIR retains its perceptual superiority.

**Slide 10: Evaluation (High Degradation)**
Moving to a high degradation scenario with a scale factor of 4, the inverse problem is significantly harder. Here, UNet starts to over-smooth the image heavily, losing most of the fine texture information.

**Slide 11: Evaluation (Severe Degradation)**
Under severe degradation—with a high scale factor and high noise—the differences become extreme. UNet outputs a blurry image, TV preserves edges but looks completely artificial, and DiffPIR tries its best to invent details, sometimes introducing artifacts.

**Slide 12: UNet Limitations**
To better understand these results, let's analyze each model's limitations. UNet is incredibly robust across all degradation levels without needing prior knowledge of the noise or scale factor. However, its reliance on the L1 loss causes it to average out high-frequency details, resulting in flat, over-smoothed textures. Additionally, it struggles with generalization, often failing to reconstruct Out-of-Distribution images it hasn't seen during training.

**Slide 13: TV Limitations**
For Total Variation, the prior forces complex natural textures into uniform color blocks. This preserves macro-edges but completely destroys photorealism, resulting in an artificial "watercolor" or staircase effect.

**Slide 14: DiffPIR Limitations**
Finally, because DiffPIR actively samples from a generative prior, it confidently invents structures. When the original information is completely lost, like in severe degradations, this conflict between the generative prior and data consistency causes strong hallucinations and unnatural grid-like artifacts.

**Slide 15: Conclusions**
In conclusion: 
- UNet offers excellent metric performance and robustness, but lacks photorealistic details. 
- TV has solid mathematical guarantees but fails on natural textures. 
- DiffPIR achieves highly realistic results, but is prone to hallucinations under extreme conditions.

**Slide 16: Thank you!**
Thank you for your attention!
