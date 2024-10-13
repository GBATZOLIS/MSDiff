# Non-Uniform Diffusion Models for Deep Generative Modeling

This repository extends Yang Song's codebase from [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS), utilizing PyTorch and PyTorch-Lightning.

We introduce **non-uniform diffusion models**, leading to **multi-scale diffusion models** that enable faster and more efficient generative modeling. By allowing different parts of the input to diffuse at varying rates, we achieve:

- **Better FID scores** than standard uniform diffusion models with equal or less training time.
- **Significantly faster sample generation**, up to 4.4× faster at 128×128 resolution.
- **A novel estimator for the conditional score function** (CMDE), performing on par with state-of-the-art methods.

**Our contributions include:**

1. **A principled framework** for training non-uniform diffusion models.
2. **Efficient multi-scale models** with improved performance and faster sampling.
3. **The Conditional Multi-Speed Diffusive Estimator (CMDE)** for conditional score estimation.
4. **A proof of consistency** for the Conditional Denoising Estimator (CDE).
5. **An empirical comparison** of methods on tasks like super-resolution and inpainting.
6. **The open-source library** MSDiff to support further research.

---

By exploring non-uniform diffusion, we aim to advance deep generative modeling with faster and more effective models.

## Instructions

All the information for every experiment is stored in configuration Python files. We used the `ml_collections` Python library for constructing the configuration files.

Once you have adjusted the relevant sections of the configuration, you can train or test the model using the following command:

```bash
python -m main.py --mode [train|test] --config [path_to_config]
```

Replace `[train|test]` with either `train` or `test`, and `[path_to_config]` with the path to your configuration file.

---

## Multi-Scale Diffusion Experiments

We have included all the configurations for the multi-scale diffusion experiments presented in our paper under the folder: `configs/vp`.

### ImageNet 128×128

- **Multiscale Model:**

  - `configs/vp/ImageNet/multiscale/resolution_128/ema/multiscale_ema.py`

- **Multiscale Model (Deep):**

  - `configs/vp/ImageNet/multiscale/resolution_128/ema/deep_multiscale/multiscale.py`

- **Vanilla (Standard Uniform Diffusion) Model:**

  - `configs/vp/ImageNet/multiscale/resolution_128/ema/vanilla_ema.py`

### CelebA-HQ 128×128

- **Multiscale Model:**

  - `configs/vp/celebA_HQ_160/new_experiments/multiscale/resolution_128/same_param_model/multiscale.py`

- **Multiscale Model (Deep):**

  - `configs/vp/celebA_HQ_160/new_experiments/multiscale/resolution_128/deeper_model/multiscale.py`

- **Vanilla (Standard Uniform Diffusion) Model:**

  - `configs/vp/celebA_HQ_160/new_experiments/vanilla/resolution_128/vanilla.py`

### Training and Testing

To train or test the multi-scale diffusion models, use the following command, replacing `[path_to_config]` with the path to the desired configuration file:

```bash
python -m main.py --mode [train|test] --config [path_to_config]
```

**Example:**

To train the multiscale model on ImageNet:

```bash
python -m main.py --mode train --config configs/vp/ImageNet/multiscale/resolution_128/ema/multiscale_ema.py
```

The multi-scale diffusion experiments use the Lightning module:

- `lightning_modules/MultiScaleSdeGenerativeModel.py`

---

## Conditional Image Generation Experiments

We have included all the configurations for the conditional generation experiments presented in our paper under the folder: `configs/ve/inverse_problems`.

### Super-Resolution

- **VS-CMDE:**

  - `configs/ve/inverse_problems/super_resolution/celebA_ours_DV_160.py`

- **CMDE:**

  - `configs/ve/inverse_problems/super_resolution/celebA_ours_NDV_160.py`

- **CDiffE:**

  - `configs/ve/inverse_problems/super_resolution/celebA_song_160.py`

- **CDE:**

  - `configs/ve/inverse_problems/super_resolution/celebA_SR3_160.py`

### Inpainting

- **VS-CMDE:**

  - `configs/ve/inverse_problems/inpainting/celebA_ours_DV.py`

- **CMDE:**

  - `configs/ve/inverse_problems/inpainting/celebA_ours_NDV.py`

- **CDiffE:**

  - `configs/ve/inverse_problems/inpainting/celebA_song.py`

- **CDE:**

  - `configs/ve/inverse_problems/inpainting/celebA_SR3.py`

### Edge to Photo Translation

- **VS-CMDE:**

  - `configs/ve/inverse_problems/image_to_image_translation/edges2shoes_ours_DV.py`

- **CMDE:**

  - `configs/ve/inverse_problems/image_to_image_translation/edges2shoes_ours_NDV.py`

- **CDiffE:**

  - `configs/ve/inverse_problems/image_to_image_translation/edges2shoes_song.py`

- **CDE:**

  - `configs/ve/inverse_problems/image_to_image_translation/edges2shoes_SR3.py`

### Training and Testing

To train or test the conditional generation models, use the following command, replacing `[path_to_config]` with the path to the desired configuration file:

```bash
python -m main.py --mode [train|test] --config [path_to_config]
```

**Example:**

To train the VS-CMDE model for super-resolution:

```bash
python -m main.py --mode train --config configs/ve/inverse_problems/super_resolution/celebA_ours_DV_160.py
```

The conditional generation experiments use the Lightning module:

- `lightning_modules/ConditionalSdeGenerativeModel.py`

---

## Citation

If you use this code for your research, please cite our paper:

[Non-Uniform Diffusion Models for Deep Generative Modeling](https://arxiv.org/pdf/2207.09786)

BibTeX:

```bibtex
@article{batzolis2022non,
  title={Non-uniform diffusion models},
  author={Batzolis, Georgios and Stanczuk, Jan and Sch{\"o}nlieb, Carola-Bibiane and Etmann, Christian},
  journal={arXiv preprint arXiv:2207.09786},
  year={2022}
}
```

---

---

Feel free to open an issue or contact us if you have any questions or need assistance with the code.
