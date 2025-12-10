## SiT — Learning Fork

This repository is a personal learning and experimentation fork based on the original SiT (Scalable Interpolant Transformers) implementation.

This fork is intended for study, code exploration and small-scale experiments. It is not an official release and the original
SiT authors and repository should be credited for the model design and pre-trained weights.



Original resources
- Paper: https://arxiv.org/pdf/2401.08740.pdf
- Original code: https://github.com/willisma/SiT




What is in this repository
- `models.py`: PyTorch implementation of SiT (patchify, adaLN, blocks, final layer).
- `train.py`: Training entrypoint (supports PyTorch DDP; designed for multi-GPU experiments).
- `sample.py`, `sample_ddp.py`: Single-GPU and multi-GPU sampling scripts (ODE / SDE samplers).
- `transport/`: ODE/SDE samplers, interpolant path implementations and utilities.

Notes and usage (learning-focused)
- Attribution: This fork builds on the original SiT repository — please cite the original paper if you use the model.
- Checkpoint compatibility: If you load upstream pre-trained weights, ensure model configuration (e.g. `num_classes`, `patch_size`) matches the checkpoint. If `num_classes` differs, you must reinitialize the label embedding (`y_embedder.embedding_table.weight`) before loading.
- Memory: Large SiT variants require multiple GPUs for training and may need substantial GPU RAM for sampling. For local experiments use smaller variants (e.g. `SiT-S/2`) and lower image sizes (256).

Quick start — sampling (single GPU)
```bash
conda env create -f environment.yml
conda activate SiT
python sample.py ODE --image-size 256 --seed 1
```

Quick start — small training (single GPU / multi-GPU)
- Single GPU: pick a small model and reduce batch size.
- Multi-GPU (example):
```bash
torchrun --nnodes=1 --nproc_per_node=4 train.py --model SiT-XL/2 --data-path /path/to/data
```

If you want help
- I can adapt `sample.py` to reduce memory usage or add a tiny example dataset and a smoke-test script.
- I can also add a short `USAGE.md` explaining how to load upstream checkpoints safely when `num_classes` differ.

License
- This repository includes the MIT license that came with the code. See `LICENSE.txt`.

— End —

