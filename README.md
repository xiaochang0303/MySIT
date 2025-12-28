USDiT
=====

USDiT is a PyTorch project for SiT-based diffusion training, ControlNet-style conditioning,
and lightweight small-sample finetuning (LoRA + lightweight ControlNet). It includes
training, sampling, and data-augmentation utilities aimed at paired image+mask datasets.

What is inside
--------------
- SiT backbone: Transformer-based diffusion model (`models.py`)
- Transport: path sampling + training losses + ODE/SDE samplers (`transport/`)
- ControlNet: full ControlSiT and lightweight ControlSiT (`controlnet.py`, `lightweight_controlnet.py`)
- LoRA injection: low-rank adapters for attention layers (`lora.py`)
- Datasets/transforms: paired image+mask pipelines (`maskdataset.py`)
- Training scripts: SiT training, ControlNet training, small-sample finetuning
- Sampling scripts: single-GPU and DDP samplers, multi-GPU controlled generation

Environment
-----------
Recommended Python: 3.10+ (3.11/3.12 OK). GPU is required for training.

Suggested dependencies (install versions that match your CUDA and PyTorch):
- torch, torchvision
- timm
- diffusers
- torchdiffeq
- numpy, pillow
- wandb (optional)
- opencv-python (optional, for Canny edges)

Data layout
-----------
PairedLayeredDataset expects:

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

For simple edge conditioning you can use a single folder of images and rely on
the Canny-based transform.

Training
--------
1) Train a base SiT model (ImageFolder):

```
torchrun --nproc_per_node=4 train.py \
  --data-path /path/to/images \
  --results-dir results \
  --model SiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --wandb
```

2) Train ControlSiT on paired image/mask data:

```
torchrun --nproc_per_node=4 train_controlnet.py \
  --data-path /path/to/data_root \
  --ckpt /path/to/pretrained_sit.pt \
  --model SiT-XL/2 \
  --image-size 256 \
  --global-batch-size 128 \
  --is-training True
```

3) Small-sample finetuning (lightweight ControlNet + optional LoRA):

```
python train_small_sample.py \
  --data-path /path/to/data_root \
  --base-ckpt /path/to/pretrained_sit.pt \
  --model SiT-XL/2 \
  --use-lora \
  --epochs 100
```

Sampling
--------
Single-GPU sampling (SiT):

```
python sample.py ODE --model SiT-XL/2 --image-size 256 --cfg-scale 4.0
```

DDP sampling (SiT):

```
torchrun --nproc_per_node=4 sample_ddp.py ODE --model SiT-XL/2 --image-size 256
```

Multi-GPU ControlNet sampling:

```
python generate_multigpu.py \
  --ckpt /path/to/controlnet_ckpt.pt \
  --data-path /path/to/data_root \
  --num-gpus 2 \
  --image-size 256
```

Augmentation
------------
Generate diverse augmented samples from masks:

```
python generate_augmented.py \
  --model-ckpt outputs/small_sample/best_controlnet.pt \
  --base-ckpt /path/to/pretrained_sit.pt \
  --mask-dir /path/to/masks \
  --output-dir augmented_data \
  --samples-per-mask 20
```

Configuration notes
-------------------
- VAE: most scripts use `stabilityai/sd-vae-ft-{ema|mse}` from diffusers.
- W&B: set `WANDB_KEY`, `ENTITY`, `PROJECT` if you enable `--wandb`.

Architecture and dependency graph
---------------------------------
See `docs/ARCHITECTURE.md` for a concise dependency graph and module overview.
