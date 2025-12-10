
## Setup

First, download and set up the repo:

```bash
git clone https://github.com/willisma/SiT.git
cd SiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate SiT
```


## Sampling [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/willisma/SiT/blob/main/run_SiT.ipynb)
![More SiT samples](visuals/visual_2.png)

**Pre-trained SiT checkpoints.** You can sample from our pre-trained SiT models with [`sample.py`](sample.py). Weights for our pre-trained SiT model will be 
automatically downloaded depending on the model you use. The script has various arguments to adjust sampler configurations (ODE & SDE), sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 256x256 SiT-XL model with default ODE setting, you can use:

```bash
python sample.py ODE --image-size 256 --seed 1
```

For convenience, our pre-trained SiT models can be downloaded directly here as well:

| SiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) | 256x256          | 2.06    | 270.27         | 119    |
<!-- | [XL/2](https://dl.fbaipublicfiles.com/SiT/models/SiT-XL-2-512x512.pt) | 512x512          | 2.62    |   252.21       | 525    | -->


**Custom SiT checkpoints.** If you've trained a new SiT model with [`train.py`](train.py) (see [below](#training-SiT)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 SiT-L/4 model with ODE sampler, run:

```bash
python sample.py ODE --model SiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

### Advanced sampler settings

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| ODE | `--atol` | `float` |  Absolute error tolerance |
|     | `--rtol` | `float` | Relative error tolenrace |   
|     | `--sampling-method` | `str` | Sampling methods (refer to [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) ) |

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| SDE | `--diffusion-form` | `str` | Form of SDE's diffusion coefficient (refer to Tab. 2 in [paper]()) |
|     | `--diffusion-norm` | `float` | Magnitude of SDE's diffusion coefficient |
|     | `--last-step` | `str` | Form of SDE's last step |
|     |               |       | None - Single SDE integration step |
|     |               |       | "Mean" - SDE integration step without diffusion coefficient |
|     |               |       | "Tweedie" - [Tweedie's denoising](https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf) step | 
|     |               |       | "Euler" - Single ODE integration step
|     | `--sampling-method` | `str` | Sampling methods |
|     |               |       | "Euler" - First order integration | 
|     |               |       | "Heun" - Second order integration | 

There are some more options; refer to [`train_utils.py`](train_utils.py) for details.

## Training SiT

We provide a training script for SiT in [`train.py`](train.py). To launch SiT-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train
```

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Then in training command add the `--wandb` flag:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --wandb
```

**Interpolant settings.** We also support different choices of interpolant and model predictions. For example, to launch SiT-XL/2 (256x256) with `Linear` interpolant and `noise` prediction: 

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --path-type Linear --prediction noise
```

**Resume training.** To resume training from custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-L/2 --data-path /path/to/imagenet/train --ckpt /path/to/model.pt
```

**Caution.** Resuming training will automatically restore both model, EMA, and optimizer states and training configs to be the same as in the checkpoint.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a SiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained SiT-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000
```

**Likelihood.** Likelihood evaluation is supported. To calculate likelihood, you can add the `--likelihood` flag to ODE sampler:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --likelihood
```

Notice that only under ODE sampler likelihood can be calculated; see [`sample_ddp.py`](sample_ddp.py) for more details and settings. 

### Enhancements
Training (and sampling) could likely be speed-up significantly by:
- [ ] using [Flash Attention](https://github.com/HazyResearch/flash-attention) in the SiT model
- [ ] using `torch.compile` in PyTorch 2.0

Basic features that would be nice to add:
- [ ] Monitor FID and other metrics
- [ ] AMP/bfloat16 support

Precision in likelihood calculation could likely be improved by:
- [ ] Uniform / Gaussian Dequantization


## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling on different platforms (TPU vs. GPU). We observed that sampling on TPU performs marginally worse than GPU (2.15 FID 
versus 2.06 in the paper).


## License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.


