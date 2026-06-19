# Compact Presentation Speech 🎤

*Here is a simple, direct speech in English. It avoids complex jargon and focuses on clearly explaining what you did in each slide.*

---

**Slide 1: Title**
Hello everyone. Today I will present our project on Super-Resolution for Computational Imaging.

**Slide 2: The Inverse Problem**
Our goal is to take a blurry, low-resolution image and reconstruct a clear, high-resolution one. This is a very difficult problem because the same blurry image could come from many different high-resolution images. In our model, we apply blur and add noise to simulate this degradation.

**Slide 3: Project Overview**
To solve this problem, we compared three different approaches:
1. A Deep Learning model called UNet.
2. A mathematical optimization method called Total Variation.
3. A Generative model called DiffPIR, based on diffusion.

**Slide 4: Dataset & Pipeline**
We used the LSUN Church dataset. We resized the images to 256x256 and tested our models on 4 different scenarios, changing the amount of blur and noise.

**Slide 5: Method 1 - UNet**
Our first method is the UNet. We trained it to directly map the blurry image back to the clear one using the L1 Loss. This teaches the network to minimize the average error, resulting in excellent noise removal but leaving the textures a bit smooth.

**Slide 6: Method 2 - Total Variation (TV)**
The second method uses pure math, without any neural network training. Total Variation relies on data consistency and a prior that favors flat colors separated by sharp edges. This preserves the macro-structure of the buildings very well.

**Slide 7: Method 3 - Generative (DiffPIR)**
Our final method is DiffPIR. It uses a pre-trained diffusion model, which already knows what real churches look like. It interleaves denoising steps with our data constraint, allowing it to actively "hallucinate" incredibly realistic textures.

**Slide 8: Evaluation (Mild Degradation)**
Here we can see the results on our first scenario, a mild degradation. If we look at the numbers, UNet dominates quantitatively. However, visually, DiffPIR looks much sharper and natural to the human eye, despite having lower metric scores. This highlights the classic perception-distortion tradeoff.

**Slide 9: Evaluation (Moderate Degradation)**
Increasing the noise slightly to 0.01, we observe a small drop in the PSNR across all methods. However, the overall visual behaviors of the three models remain consistent with the previous scenario.

**Slide 10: Evaluation (High Degradation)**
Moving to a much stronger degradation with a scale factor of 4, the inverse problem becomes significantly harder. UNet starts to over-smooth the image heavily, losing almost all fine textures.

**Slide 11: Evaluation (Severe Degradation)**
When the input image is heavily degraded with both a scale factor of 4 and high noise, the differences become extreme. UNet completely collapses and blurs the image. TV keeps the macro-structure but looks artificial. DiffPIR tries to invent details but can introduce heavy artifacts.

**Slide 12: UNet Limitations**
To better understand these flaws, let's look at the limitations of each model. Since UNet minimizes the L1 loss, it suffers from "regression to the mean", averaging out all high-frequency details. Moreover, if we test it on data it hasn't seen during training, it fails dramatically.

**Slide 13: TV Limitations**
For Total Variation, the assumption that images are "piecewise constant" causes a strong staircase effect. As you can see, complex natural textures are merged into uniform blocks of color, giving the image an unnatural "watercolor" appearance.

**Slide 14: DiffPIR Limitations**
Finally, because DiffPIR actively samples from a generative prior, it confidently invents structures that were never there. Under severe degradation, the conflict between its prior and the real data causes strong hallucinations and unnatural grid-like patterns.

**Slide 15: Conclusions**
In conclusion: 
- UNet is fast and mathematically accurate, but lacks structural robustness. 
- TV has solid mathematical guarantees but is unsuitable for natural textures. 
- DiffPIR recreates highly realistic images, but produces hallucinations and artifacts under severe degradation.

Thank you for your attention!
