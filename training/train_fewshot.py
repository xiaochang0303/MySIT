"""
小样本 ControlNet 微调训练脚本

针对 <100 张图片的小样本场景优化，包含:
- 强数据增强
- 防过拟合机制 (早停、验证集监控、正则化)
- 学习率 warmup + cosine decay
- 控制强度随机化
- 更高的 EMA decay

Usage:
    # 单 GPU (用于小样本)
    python training/train_fewshot.py \
        --data-path sample_data \
        --ckpt /path/to/pretrained_sit.pt \
        --epochs 200 \
        --batch-size 4

    # 多 GPU
    torchrun --nproc_per_node=2 training/train_fewshot.py \
        --data-path sample_data \
        --ckpt /path/to/pretrained_sit.pt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import math
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import DistributedSampler

from torchvision.utils import make_grid, save_image

from models import SiT_models, ControlSiT, LightweightControlSiT
from models import inject_lora, get_lora_parameters, count_lora_parameters
from utils import find_model
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from training.train_utils import parse_transport_args, log_training_config
from datasets import PairedLayeredDataset, PairedFlatDataset, PairedTransform

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9995):
    """EMA 更新，小样本场景使用更高的 decay"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir, rank=0):
    if rank == 0:
        os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class StrongPairedTransform:
    """
    强数据增强 Transform，适用于小样本场景

    包含:
    - 随机裁剪 (而非固定 center crop)
    - 随机翻转
    - 颜色抖动 (仅对图像)
    - 随机旋转
    - 随机缩放
    """
    def __init__(self, image_size, is_training=True,
                 color_jitter=0.2, rotation_range=15, scale_range=(0.9, 1.1)):
        self.image_size = image_size
        self.is_training = is_training
        self.color_jitter = color_jitter
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        # 基础 transform
        self.base_transform = PairedTransform(image_size, is_training)

    def __call__(self, img, mask):
        import torchvision.transforms.functional as TF
        import random

        if not self.is_training:
            return self.base_transform(img, mask)

        # 1. 随机旋转 (同步)
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img = TF.rotate(img, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        # 2. 随机缩放 + 裁剪 (同步)
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            new_size = int(min(img.size) * scale)
            if new_size >= self.image_size:
                # 缩放
                img = TF.resize(img, new_size)
                mask = TF.resize(mask, new_size, interpolation=Image.NEAREST)

                # 随机裁剪
                i, j, h, w = self._get_random_crop_params(img, self.image_size)
                img = TF.crop(img, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

        # 3. 应用基础 transform (包含 resize、flip、normalize)
        img, mask = self.base_transform(img, mask)

        # 4. 颜色抖动 (仅对图像，在 tensor 上操作)
        if self.color_jitter > 0 and random.random() > 0.5:
            # 简单的亮度/对比度调整
            brightness = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            contrast = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            img = img * contrast + (brightness - 1)
            img = torch.clamp(img, -1, 1)

        return img, mask

    def _get_random_crop_params(self, img, output_size):
        """获取随机裁剪参数"""
        import random
        w, h = img.size
        th, tw = output_size, output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine 学习率调度器，带 warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def split_dataset(dataset, val_ratio=0.1, seed=42):
    """划分训练集和验证集"""
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main(args):
    # 检测是否使用分布式
    use_ddp = 'RANK' in os.environ

    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建实验目录
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        ctrl_tag = "lightweight" if args.use_lightweight else "controlnet"
        lora_tag = f"-lora{args.lora_rank}" if args.lora_rank > 0 else ""
        experiment_name = f'{experiment_index:03d}-fewshot-{ctrl_tag}{lora_tag}-{model_string_name}'
        experiment_dir = f'{args.results_dir}/{experiment_name}'
        ckpt_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        experiment_dir = None
        ckpt_dir = None

    logger = create_logger(experiment_dir if rank == 0 else '/tmp', rank)

    if rank == 0:
        logger.info(f"Experiment: {experiment_dir}")
        log_training_config(logger, args, title="Few-Shot Training Configuration")

    # 构建模型
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # 加载预训练权重
    if args.ckpt is not None:
        state_dict = find_model(args.ckpt)
        if "ema" in state_dict:
            if "y_embedder.embedding_table.weight" in state_dict["ema"]:
                del state_dict["ema"]["y_embedder.embedding_table.weight"]
            if "pos_embed" in state_dict["ema"]:
                del state_dict["ema"]["pos_embed"]
            base.load_state_dict(state_dict["ema"], strict=False)
        else:
            base.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {args.ckpt}")

    # 创建 ControlNet
    if args.use_lightweight:
        control_model = LightweightControlSiT(
            base,
            rank=args.light_rank,
            shared_depth=args.light_shared_depth,
            freeze_base=True,
            noise_scale=args.noise_scale,
        )
    else:
        control_model = ControlSiT(base, freeze_base=True)

    control_model = control_model.to(device)

    # 可选: 注入 LoRA
    lora_params = []
    if args.lora_rank > 0:
        inject_lora(base, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        lora_params = get_lora_parameters(base)
        count_lora_parameters(base)

    # EMA
    ema = deepcopy(control_model).to(device)
    requires_grad(ema, False)

    # 收集可训练参数
    trainable_params = [p for p in control_model.parameters() if p.requires_grad]
    trainable_params.extend(lora_params)

    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # 优化器
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Transport
    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight,
        args.train_eps, args.sample_eps
    )
    sampler = Sampler(transport)

    # VAE
    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()

    # 数据集
    if args.strong_augment:
        transform = StrongPairedTransform(args.image_size, is_training=True)
    else:
        transform = PairedTransform(args.image_size, is_training=True)

    # 根据数据目录结构选择数据集类
    if os.path.exists(os.path.join(args.data_path, 'images')):
        # 扁平结构: data_path/images/, data_path/masks/
        full_dataset = PairedFlatDataset(
            args.data_path, transform=transform, class_label=args.fixed_class_id
        )
    else:
        # 分层结构: data_path/class_name/image/, data_path/class_name/mask/
        full_dataset = PairedLayeredDataset(args.data_path, transform=transform)

    # 划分训练/验证集
    if args.val_ratio > 0:
        train_dataset, val_dataset = split_dataset(full_dataset, args.val_ratio, args.global_seed)
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info(f"Train: {len(train_dataset)} (no validation)")

    # DataLoader
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    # 学习率调度器
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    # 早停
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-5) if args.use_early_stop else None

    # DDP 包装
    if use_ddp:
        control_model = DDP(control_model, device_ids=[device])

    # 初始化 EMA
    update_ema(ema, control_model.module if use_ddp else control_model, decay=0)
    control_model.train()
    ema.eval()

    # 训练循环
    train_steps = 0
    best_val_loss = float('inf')
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    # 用于采样的固定噪声
    n_sample = min(args.batch_size, 4)
    zs = torch.randn(n_sample, 4, latent_size, latent_size, device=device)

    logger.info(f"Training for {args.epochs} epochs ({total_steps} steps)...")

    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        for batch_data in train_loader:
            (img, mask), y, _ = batch_data
            img = img.to(device)
            mask = mask.to(device)
            y = y.to(device)

            if args.fixed_class_id >= 0:
                y = torch.full_like(y, args.fixed_class_id)

            # VAE 编码
            with torch.no_grad():
                x = vae.encode(img).latent_dist.sample().mul_(0.18215)

                if mask.shape[1] == 1:
                    mask_rgb = mask.repeat(1, 3, 1, 1)
                else:
                    mask_rgb = mask
                ctrl = vae.encode(mask_rgb).latent_dist.sample().mul_(0.18215)

            # 训练时随机化控制强度 (防过拟合)
            if args.random_ctrl_strength:
                ctrl_strength = np.random.uniform(0.8, 1.2)
            else:
                ctrl_strength = 1.0

            model_kwargs = dict(y=y, control=ctrl, control_strength=ctrl_strength)

            # 计算 loss
            loss_dict = transport.training_losses(control_model, x, model_kwargs)
            loss = loss_dict['loss'].mean()

            # 反向传播
            opt.zero_grad()
            loss.backward()

            # 梯度裁剪
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            opt.step()
            scheduler.step()

            # EMA 更新
            update_ema(ema, control_model.module if use_ddp else control_model, decay=args.ema_decay)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # 日志
            if train_steps % args.log_every == 0:
                avg_loss = running_loss / log_steps
                lr = scheduler.get_last_lr()[0]
                steps_per_sec = log_steps / (time() - start_time)

                logger.info(f"Step {train_steps:06d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {steps_per_sec:.1f} steps/s")

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # 采样
            if train_steps % args.sample_every == 0 and rank == 0:
                logger.info("Generating samples...")
                with torch.no_grad():
                    sample_fn = sampler.sample_ode()

                    # 使用当前 batch 的 control
                    ctrl_sample = ctrl[:n_sample]

                    if args.cfg_scale > 1.0:
                        z_in = torch.cat([zs, zs], dim=0)
                        y_sample = y[:n_sample]
                        y_null = torch.full((n_sample,), args.num_classes, device=device)
                        y_in = torch.cat([y_sample, y_null], dim=0)
                        model_fn = ema.forward_with_cfg
                        model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)
                    else:
                        z_in = zs
                        y_in = y[:n_sample]
                        model_fn = ema.forward
                        model_kwargs = dict(y=y_in)

                    samples = sample_fn(z_in, model_fn, control=ctrl_sample, **model_kwargs)[-1]

                    if args.cfg_scale > 1.0:
                        samples, _ = samples.chunk(2, dim=0)

                    samples = vae.decode(samples / 0.18215).sample.clamp(-1, 1)

                    # 可视化
                    img_vis = (img[:n_sample].cpu() + 1) * 0.5
                    mask_vis = (mask[:n_sample].cpu() + 1) * 0.5
                    if mask_vis.shape[1] == 1:
                        mask_vis = mask_vis.repeat(1, 3, 1, 1)
                    gen_vis = (samples.cpu() + 1) * 0.5

                    tiles = []
                    for i in range(n_sample):
                        tiles.extend([img_vis[i], mask_vis[i], gen_vis[i]])

                    grid = make_grid(tiles, nrow=3, padding=2)
                    save_image(grid, f'{ckpt_dir}/step_{train_steps:06d}.png')

                logger.info("Sampling done.")

        # Epoch 结束: 验证
        if val_loader is not None and rank == 0:
            val_loss = 0.0
            val_steps = 0
            control_model.eval()

            with torch.no_grad():
                for batch_data in val_loader:
                    (img, mask), y, _ = batch_data
                    img = img.to(device)
                    mask = mask.to(device)
                    y = y.to(device)

                    if args.fixed_class_id >= 0:
                        y = torch.full_like(y, args.fixed_class_id)

                    x = vae.encode(img).latent_dist.sample().mul_(0.18215)
                    if mask.shape[1] == 1:
                        mask_rgb = mask.repeat(1, 3, 1, 1)
                    else:
                        mask_rgb = mask
                    ctrl = vae.encode(mask_rgb).latent_dist.sample().mul_(0.18215)

                    model_kwargs = dict(y=y, control=ctrl)
                    loss_dict = transport.training_losses(
                        control_model.module if use_ddp else control_model, x, model_kwargs
                    )
                    val_loss += loss_dict['loss'].mean().item()
                    val_steps += 1

            control_model.train()
            val_loss /= max(val_steps, 1)
            logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": (control_model.module if use_ddp else control_model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": args,
                }
                torch.save(checkpoint, f'{ckpt_dir}/best.pt')
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")

            # 早停检查
            if early_stopping is not None and early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # 定期保存
        if (epoch + 1) % args.save_every == 0 and rank == 0:
            checkpoint = {
                "model": (control_model.module if use_ddp else control_model).state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(checkpoint, f'{ckpt_dir}/epoch_{epoch:04d}.pt')

    # 保存最终模型
    if rank == 0:
        checkpoint = {
            "model": (control_model.module if use_ddp else control_model).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "epoch": args.epochs,
            "args": args,
        }
        torch.save(checkpoint, f'{ckpt_dir}/final.pt')
        logger.info(f"Training complete. Final model saved to {ckpt_dir}/final.pt")

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")

    # 模型
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, required=True, help="预训练 SiT checkpoint")
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="ema")

    # ControlNet 类型
    parser.add_argument("--use-lightweight", action="store_true", help="使用轻量级 ControlNet")
    parser.add_argument("--light-rank", type=int, default=32)
    parser.add_argument("--light-shared-depth", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=0.05, help="训练时噪声注入")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0=禁用)")
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.9995, help="EMA decay (小样本用更高值)")

    # 防过拟合
    parser.add_argument("--strong-augment", action="store_true", help="使用强数据增强")
    parser.add_argument("--random-ctrl-strength", action="store_true", help="随机化控制强度")
    parser.add_argument("--use-early-stop", action="store_true", help="启用早停")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")

    # 采样
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--fixed-class-id", type=int, default=0)

    # 日志
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=42)

    # Transport
    parse_transport_args(parser)

    args = parser.parse_args()
    main(args)
