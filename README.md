# Conditional Image Generation with Score-Based Diffusion Models

This repository is an extension of the code base provided by Yang Song for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS).

We have extended the code to support multi speed/sde diffusion. Multi speed diffusion opens the avenue for further research in conditional generation, learning in multiple scales and represenation learning using the score-based diffusion framework.

In this paper, we use multi speed diffusion to derive the CMDE and VS-CMDE estimators of conditional score. Those estimators are used for conditional image generation. We also provide the code for training conditional score models using the conditional denoising estimator (CDE).
