# Conditional Image Generation with Score-Based Diffusion Models

This repository is an extension of the code base provided by Yang Song for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS). The code depends on pytorch and pytorch-lightning.

We have extended the code to support multi speed/sde diffusion. Multi speed diffusion opens the avenue for further research in conditional generation and hierarchical represenation learning using the score-based diffusion framework.

In this paper, we use multi speed diffusion to derive the CMDE and VS-CMDE estimators of conditional score. Those estimators are used for conditional image generation. We also provide the code for training conditional score models using the conditional denoising estimator (CDE).

Instructions: 

All the information for every experiment is stored in configurational python files. We used the ml_collections python library for constructing the configurational files. Once you have re-written the relevant sections of the configuration you can simply train or test the configuration using the following command:

python -m main.py --mode train or test --config path_to_config

We have included all the configuration files for all the experiments presented in the paper:
All of the configuration files are under the folder: configs/ve/inverse_problems

For super-resolution: 

VS-CMDE: configs/ve/inverse_problems/super_resolution/celebA_ours_DV_160.py \
CMDE: configs/ve/inverse_problems/super_resolution/celebA_ours_NDV_160.py \
CDiffE: configs/ve/inverse_problems/super_resolution/celebA_song_160.py \
CDE: configs/ve/inverse_problems/super_resolution/celebA_SR3_160.py

For inpainting: 

VS-CMDE: configs/ve/inverse_problems/inpainting/celebA_ours_DV.py \
CMDE: configs/ve/inverse_problems/inpainting/celebA_ours_NDV.py \
CDiffE: configs/ve/inverse_problems/inpainting/celebA_song.py \
CDE: configs/ve/inverse_problems/inpainting/celebA_SR3.py

For edge to photo translation: 

VS-CMDE: configs/ve/inverse_problems/image_to_image_translation/edges2shoes_ours_DV.py \
CMDE: configs/ve/inverse_problems/image_to_image_translation/edges2shoes_ours_NDV.py \
CDiffE: configs/ve/inverse_problems/image_to_image_translation/edges2shoes_song.py \
CDE: configs/ve/inverse_problems/image_to_image_translation/edges2shoes_SR3.py
