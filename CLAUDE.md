# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USDiT is a PyTorch project for training SiT-based (Scalable Interpolant Transformer) diffusion models with ControlNet-style conditioning and lightweight small-sample finetuning capabilities (LoRA + lightweight ControlNet). The project focuses on paired image+mask datasets for controlled image generation.

## Common Commands

### Training

**Train base SiT model (ImageFolder-based):**
```bash
torchrun --nproc_per_node=4 train.py \
  --data-path /path/to/images \
  --results-dir results \
  --model SiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --wandb
```

**Train ControlSiT on paired image/mask data:**
```bash
torchrun --nproc_per_node=4 train_controlnet.py \
  --data-path /path/to/data_root \
  --ckpt /path/to/pretrained_sit.pt \
  --model SiT-XL/2 \
  --image-size 256 \
  --global-batch-size 128 \
  --is-training True
```

**Small-sample finetuning (lightweight ControlNet + optional LoRA):**
```bash
python train_small_sample.py \
  --data-path /path/to/data_root \
  --base-ckpt /path/to/pretrained_sit.pt \
  --model SiT-XL/2 \
  --use-lora \
  --epochs 100
```

### Sampling

**Single-GPU sampling (SiT):**
```bash
python sample.py ODE --model SiT-XL/2 --image-size 256 --cfg-scale 4.0
```

**DDP sampling (SiT):**
```bash
torchrun --nproc_per_node=4 sample_ddp.py ODE --model SiT-XL/2 --image-size 256
```

**Multi-GPU ControlNet sampling:**
```bash
python generate_multigpu.py \
  --ckpt /path/to/controlnet_ckpt.pt \
  --data-path /path/to/data_root \
  --num-gpus 2 \
  --image-size 256
```

### Data Augmentation

**Generate diverse augmented samples from masks:**
```bash
python generate_augmented.py \
  --model-ckpt outputs/small_sample/best_controlnet.pt \
  --base-ckpt /path/to/pretrained_sit.pt \
  --mask-dir /path/to/masks \
  --output-dir augmented_data \
  --samples-per-mask 20
```

## Architecture

### Core Modules

**models.py:**
- Core SiT (Scalable Interpolant Transformer) backbone
- Transformer-based diffusion model with adaLN-Zero conditioning
- Includes TimestepEmbedder, LabelEmbedder for temporal and class conditioning
- Multiple model variants: SiT-XL/2, SiT-L/2, SiT-B/2, SiT-S/2 (and /4, /8 patch sizes)

**controlnet.py / lightweight_controlnet.py:**
- ControlNet variants for conditioning SiT on mask/edge inputs
- Full ControlSiT: duplicates SiT blocks and injects residuals
- Lightweight ControlSiT: uses lightweight adapter blocks (ControlAdapter) with Zero-initialized projection layers
- Both maintain frozen base model weights and train only control components

**lora.py:**
- Low-rank adaptation (LoRA) for efficient finetuning
- Injects low-rank matrices into attention layers
- Used for small-sample finetuning scenarios

**transport/ (transport framework):**
- `transport.py`: Core Transport class for path sampling and training losses
- `path.py`: Path types (Linear, GVP, VP) for diffusion trajectories
- `integrators.py`: ODE/SDE samplers for generation
- Supports multiple prediction types: velocity (default), score, noise
- Supports multiple loss weightings: uniform, velocity-weighted, likelihood-weighted

**maskdataset.py:**
- `PairedLayeredDataset`: Loads paired image+mask data from nested folder structure
- `ImageWithCanny`: Transform that generates Canny edge maps as control signals
- `PairedTransform`: Synchronized transforms for image+mask pairs
- Expects data layout: `data_root/class_name/{image,mask}/filename.ext`

### Training Flow

1. Images and control inputs (masks/edges) are loaded via dataset classes
2. VAE (`diffusers.AutoencoderKL`) encodes images and controls to latent space
3. Transport samples path points (t) and computes target predictions (velocity/score/noise)
4. SiT or ControlSiT predicts based on latent, timestep, and class/control conditioning
5. Loss is computed and backpropagated; EMA copies are maintained for sampling

### Sampling Flow

1. Start with random noise in latent space
2. Sampler (ODE or SDE from `transport/`) generates trajectory using model predictions
3. Model uses `forward_with_cfg` for classifier-free guidance if cfg_scale > 1.0
4. VAE decodes final latents back to images

## Key Configuration Details

**VAE:** Most scripts use `stabilityai/sd-vae-ft-{ema|mse}` from diffusers. The latent space is 8x downsampled, so image_size must be divisible by 8.

**W&B Integration:** Set environment variables `WANDB_KEY`, `ENTITY`, `PROJECT` if using `--wandb` flag.

**DDP Training:** All training scripts use PyTorch DDP. Launch with `torchrun --nproc_per_node=N`.

**Model Variants:**
- Sizes: S (small), B (base), L (large), XL (extra-large)
- Patch sizes: /2, /4, /8 (smaller = more patches = more compute)
- Example: SiT-XL/2 has depth=28, hidden_size=1152, patch_size=2, num_heads=16

**Classifier-Free Guidance:** Enabled via `cfg_scale` parameter. During training, labels are randomly dropped with `class_dropout_prob=0.1`. During sampling, use cfg_scale > 1.0 (typically 4.0) for better quality.

## Data Layout

PairedLayeredDataset expects:
```
data_root/
  class_a/
    image/
      0001.png
      0002.png
    mask/
      0001.png
      0002.png
  class_b/
    image/
    mask/
```

For Canny edge conditioning, you can use a single folder of images with `ImageWithCanny` transform.

## Transport System

The transport framework abstracts the diffusion process:
- **Path types**: Linear (straight interpolation), GVP, VP (variance preserving)
- **Prediction types**: velocity (default, most stable), score, noise
- **Loss weights**: uniform, velocity-weighted, likelihood-weighted
- Use `parse_transport_args(parser)` in training scripts to add transport-related CLI arguments
- Use `create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)` to instantiate

## ControlNet Architecture

ControlNet variants inject control information (masks, edges) into the base SiT model:
- **Full ControlSiT**: Clones SiT blocks into a control branch, processes control input, adds residuals
- **Lightweight ControlSiT**: Uses compact ControlAdapter with MLP blocks and Zero-initialized outputs
- Both freeze the base SiT weights and train only the control components
- Control embedder uses PatchEmbed to convert control images to tokens

## LoRA Integration

LoRA enables efficient finetuning with minimal parameters:
- Injects low-rank matrices (A, B) into attention layers: W' = W + BA
- Rank typically 4-16, alpha parameter controls scaling
- Only LoRA parameters are trained, base model frozen
- Combine with lightweight ControlNet for small-sample finetuning scenarios
