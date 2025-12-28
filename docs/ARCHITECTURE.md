Architecture Overview
=====================

This document describes how core modules depend on each other and how data
flows through training and sampling.

Dependency Graph (Modules)
--------------------------
Mermaid graph of the main Python modules:

```mermaid
graph TD
  train.py --> models.py
  train.py --> transport/
  train.py --> train_utils.py
  train.py --> download.py
  train.py --> wandb_utils.py
  train.py --> diffusers
  train_controlnet.py --> models.py
  train_controlnet.py --> controlnet.py
  train_controlnet.py --> maskdataset.py
  train_controlnet.py --> transport/
  train_controlnet.py --> train_utils.py
  train_controlnet.py --> download.py
  train_controlnet.py --> diffusers

  train_small_sample.py --> models.py
  train_small_sample.py --> lightweight_controlnet.py
  train_small_sample.py --> lora.py
  train_small_sample.py --> maskdataset.py
  train_small_sample.py --> transport/

  sample.py --> models.py
  sample.py --> transport/
  sample.py --> download.py
  sample.py --> diffusers
  sample_ddp.py --> models.py
  sample_ddp.py --> transport/
  sample_ddp.py --> download.py
  sample_ddp.py --> diffusers

  generate_multigpu.py --> controlnet.py
  generate_multigpu.py --> models.py
  generate_multigpu.py --> maskdataset.py
  generate_multigpu.py --> transport/
  generate_multigpu.py --> download.py
  generate_multigpu.py --> diffusers

  generate_augmented.py --> lightweight_controlnet.py
  generate_augmented.py --> lora.py
  generate_augmented.py --> diverse_sampler.py
  generate_augmented.py --> transport/
  generate_augmented.py --> diffusers

  controlnet.py --> models.py
  lightweight_controlnet.py --> models.py
  lora.py --> models.py
  diverse_sampler.py --> transport/
```

Data Flow (Training)
-------------------
- Dataset: images (+ masks/edges) loaded by `maskdataset.py` or `ImageFolder`.
- VAE: images and control inputs are encoded to latents by `diffusers.AutoencoderKL`.
- Transport: `Transport.training_losses` samples path points and computes loss targets.
- Model: SiT or ControlSiT predicts velocity/score/noise depending on configuration.
- Optimization: optimizers update trainable parameters; EMA copies are maintained for sampling.

Data Flow (Sampling)
-------------------
- Sampler: `Sampler.sample_ode` or `Sampler.sample_sde` generates latent trajectories.
- Model: forward or `forward_with_cfg` predicts drift or denoising outputs.
- VAE: decodes final latents back to images.

Notes
-----
- `transport/` is the core for path planning, loss types, and ODE/SDE integration.
- ControlNet variants inject residuals into SiT blocks to condition on masks/edges.
