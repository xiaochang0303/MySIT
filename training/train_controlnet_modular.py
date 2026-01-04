import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import inspect
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import SiT_models, ControlSiT, LightweightControlSiT
from models import inject_lora, get_lora_parameters, count_lora_parameters
from utils import find_model
import utils.wandb_utils as wandb_utils
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from training.train_utils import parse_transport_args, log_training_config
from datasets import PairedLayeredDataset, PairedTransform

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format=f'[\x1b[34m%%(asctime)s]\x1b[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
            self.paths.extend(glob(os.path.join(root, ext)))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img, edge = self.transform(img)
        else:
            img, edge = img, None

        return (img, edge), 0


def _get_signature_kwargs(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return None
    return set(sig.parameters.keys())


def _filter_model_kwargs(model, model_kwargs, fn_name="forward"):
    if not model_kwargs:
        return {}
    target = model.module if hasattr(model, "module") else model
    fn = getattr(target, fn_name, None)
    if fn is None:
        return model_kwargs
    allowed = _get_signature_kwargs(fn)
    if allowed is None:
        return model_kwargs
    return {k: v for k, v in model_kwargs.items() if k in allowed}


def build_base_model(args, device, logger):
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    if args.ckpt is not None:
        state_dict = find_model(args.ckpt)
        if "ema" in state_dict:
            if "y_embedder.embedding_table.weight" in state_dict["ema"]:
                del state_dict["ema"]["y_embedder.embedding_table.weight"]
            if "pos_embed" in state_dict["ema"]:
                del state_dict["ema"]["pos_embed"]
            missing, unexpected = base.load_state_dict(state_dict["ema"], strict=False)
            if missing:
                logger.info(f"[INFO] missing keys: {len(missing)}")
            if unexpected:
                logger.info(f"[INFO] unexpected keys: {len(unexpected)}")
        else:
            base.load_state_dict(state_dict, strict=False)

    return base.to(device), latent_size


def _parse_lora_qkv(value):
    if not value:
        return False, False, False
    tokens = {v.strip().lower() for v in value.split(",") if v.strip()}
    return ("q" in tokens), ("k" in tokens), ("v" in tokens)


def apply_lora_if_requested(base, control_model, args, logger):
    if args.lora_rank <= 0:
        return None, []

    if args.lora_only:
        requires_grad(control_model, False)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    enable_qkv = _parse_lora_qkv(args.lora_qkv)
    lora_layers = inject_lora(
        base,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
        enable_qkv=enable_qkv,
    )
    lora_params = get_lora_parameters(base)

    if args.lora_only:
        for p in lora_params:
            p.requires_grad = True

    count_lora_parameters(base)
    logger.info(f"[LoRA] trainable params: {sum(p.numel() for p in lora_params):,}")
    return lora_layers, lora_params


def build_control_model(args, base, logger):
    if args.control_type == "controlnet":
        control_model = ControlSiT(base, freeze_base=not args.unfreeze_base, cfg_channels=args.cfg_channels)
    elif args.control_type == "lightweight":
        control_model = LightweightControlSiT(
            base,
            rank=args.light_rank,
            shared_depth=args.light_shared_depth,
            freeze_base=not args.unfreeze_base,
            noise_scale=args.light_noise_scale,
            cfg_channels=args.cfg_channels,
        )
    elif args.control_type == "none":
        control_model = base
    else:
        raise ValueError(f"Unknown control type: {args.control_type}")

    trainable = sum(p.numel() for p in control_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")
    return control_model


def select_trainable_params(control_model, base, args, lora_params):
    if args.lora_only and lora_params:
        return lora_params
    return [p for p in control_model.parameters() if p.requires_grad]


def main(args):
    assert torch.cuda.is_available(), 'Training currently requires at least one GPU.'

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_bs = args.global_batch_size // dist.get_world_size()

    # Folders & logger
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        lora_tag = f"-lora{args.lora_rank}" if args.lora_rank > 0 else ""
        experiment_name = f'{experiment_index:03d}-control-{args.control_type}{lora_tag}-{model_string_name}-edges'
        experiment_dir = f'{args.results_dir}/{experiment_name}'
        ckpt_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f'Experiment dir: {experiment_dir}')
        log_training_config(logger, args, title="Modular ControlNet Training Configuration")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Model backbone
    base, latent_size = build_base_model(args, device, logger)
    control_model = build_control_model(args, base, logger).to(device)
    lora_layers, lora_params = apply_lora_if_requested(base, control_model, args, logger)

    ema = deepcopy(control_model).to(device)
    requires_grad(ema, flag=False)

    params = select_trainable_params(control_model, base, args, lora_params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)

    # Transport / sampler
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)

    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()

    # Data
    transform = PairedTransform(args.image_size, is_training=True)  # 训练脚本始终启用数据增强
    if args.single_folder:
        dataset = FlatFolderDataset(args.data_path, transform=transform)
    else:
        dataset = PairedLayeredDataset(args.data_path, transform=transform)

    ddp_sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset, batch_size=local_bs, shuffle=False, sampler=ddp_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    logger.info(f'Dataset contains {len(dataset)} images ({args.data_path}).')

    update_ema(ema, control_model, decay=0)  # copy
    control_model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    use_cfg = args.cfg_scale > 1.0
    n = local_bs

    if args.fixed_class_id >= 0:
        ys_sample = torch.full((n,), args.fixed_class_id, device=device, dtype=torch.long)
    else:
        ys_sample = torch.randint(args.num_classes, size=(n,), device=device)

    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    control_model = DDP(control_model, device_ids=[rank])

    logger.info(f'Training for {args.epochs} epochs...')

    for epoch in range(args.epochs):
        ddp_sampler.set_epoch(epoch)
        logger.info(f'Beginning epoch {epoch}...')
        for (img, edge), y, _ in loader:
            img = img.to(device)
            edge = edge.to(device)
            y = y.to(device)

            if args.fixed_class_id >= 0:
                y = torch.full_like(y, args.fixed_class_id)

            with torch.no_grad():
                x = vae.encode(img).latent_dist.sample().mul_(0.18215)

                if edge.shape[1] == 1:
                    edge_rgb = edge.repeat(1, 3, 1, 1)
                else:
                    edge_rgb = edge

                ctrl = vae.encode(edge_rgb).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y, control=ctrl, control_strength=args.control_strength)
            model_kwargs = _filter_model_kwargs(control_model, model_kwargs, fn_name="forward")

            loss_dict = transport.training_losses(control_model, x, model_kwargs)
            loss = loss_dict['loss'].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            update_ema(ema, control_model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()

                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(
                    f'Train Steps: {train_steps:07d} Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}'
                )
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps
                    )

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": control_model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    ckpt_path = f'{ckpt_dir}/{train_steps:07d}.pt'
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples (with control)...")
                with torch.no_grad():
                    sample_fn = sampler.sample_ode()
                    ys = y
                    z = zs
                    if use_cfg:
                        z = torch.cat([z, z], dim=0)
                        y_null = torch.tensor([args.num_classes] * n, device=device)
                        ys = torch.cat([ys, y_null], dim=0)
                        model_fn = ema.forward_with_cfg
                        model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale, control_strength=args.control_strength)
                    else:
                        model_fn = ema.forward
                        model_kwargs = dict(y=ys, control_strength=args.control_strength)

                    model_kwargs = _filter_model_kwargs(ema, model_kwargs, fn_name=model_fn.__name__)

                    if args.no_sample_control:
                        ctrl_latents = None
                    else:
                        # 当使用 CFG 时，control 需要复制以匹配 z 的维度
                        # forward_with_cfg 内部会处理 control 的对齐，所以这里只需要传入原始 ctrl
                        ctrl_latents = ctrl[:n]  # 使用当前 batch 的 control

                    samples = sample_fn(z, model_fn, control=ctrl_latents, **model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)

                    samples = vae.decode(samples / 0.18215).sample.clamp(-1, 1)
                    if args.wandb:
                        wandb_utils.log_image(samples, train_steps)

                    img_vis = (img.float().detach().cpu() + 1) * 0.5
                    edge_vis = edge.float().detach().cpu()
                    edge_vis = (edge_vis + 1) * 0.5

                    if edge_vis.shape[1] == 1:
                        edge_vis = edge_vis.repeat(1, 3, 1, 1)

                    gen_vis = (samples.float().detach().cpu() + 1) * 0.5

                    n = len(img_vis)
                    tiles = []
                    for i in range(n):
                        tiles.extend([img_vis[i], edge_vis[i], gen_vis[i]])

                    grid = make_grid(tiles, nrow=3, padding=2)

                    if dist.get_rank() == 0:
                        out_png = os.path.join(ckpt_dir, f'step_{train_steps:07d}.png')
                        save_image(grid, out_png)

                logger.info("Sampling done.")

    control_model.eval()
    logger.info('Done!')
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="ema")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")

    # Control and LoRA options
    parser.add_argument("--ckpt", type=str, default=None, help="Path to pretrained SiT checkpoint (.pt)")
    parser.add_argument("--control-type", type=str, choices=["controlnet", "lightweight", "none"], default="controlnet")
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-base", action="store_true", help="Whether to unfreeze the base SiT model.")
    parser.add_argument("--no-sample-control", action="store_true", help="Disable control signals during sampling.")
    parser.add_argument("--single-folder", action="store_true")

    # Lightweight ControlNet options
    parser.add_argument("--light-rank", type=int, default=32)
    parser.add_argument("--light-shared-depth", type=int, default=4)
    parser.add_argument("--light-noise-scale", type=float, default=0.0)

    # CFG options
    parser.add_argument("--cfg-channels", type=str, choices=["first3", "all"], default="first3",
                        help="CFG channel mode: 'first3' for original SiT behavior, 'all' for standard CFG on all latent channels.")

    # LoRA options
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", type=str, default="qkv,proj")
    parser.add_argument("--lora-qkv", type=str, default="q,v", help="Comma list of LoRA QKV targets.")
    parser.add_argument("--lora-only", action="store_true", help="Train only LoRA parameters.")

    # Single-class convenience
    parser.add_argument("--fixed-class-id", type=int, default=-1)

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
