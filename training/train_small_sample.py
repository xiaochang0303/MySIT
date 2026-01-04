"""
小样本超声图像生成训练示例

展示如何使用 LoRA + 轻量级 ControlNet 进行小样本微调

Usage:
    # 单 GPU 训练
    python train_small_sample.py --data-path /path/to/dataset --base-ckpt pretrained.pt

    # 多 GPU 训练
    torchrun --nproc_per_node=2 train_small_sample.py --data-path /path/to/dataset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# 本地模块
from models import SiT_models, LightweightControlSiT
from models.lightweight_controlnet import count_parameters
from models import inject_lora, count_lora_parameters
from models.lora import save_lora_weights, load_lora_weights
from transport import create_transport
from datasets import PairedLayeredDataset, PairedTransform


def create_model(args):
    """创建模型"""
    # 1. 加载预训练基座
    print(f"Loading base model: {args.model}")
    base = SiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        learn_sigma=True,
    )
    
    if args.base_ckpt:
        print(f"Loading base checkpoint: {args.base_ckpt}")
        state_dict = torch.load(args.base_ckpt, map_location="cpu")
        base.load_state_dict(state_dict)
    
    # 2. 包装为轻量级 ControlNet
    print("Creating LightweightControlSiT...")
    model = LightweightControlSiT(
        base=base,
        rank=args.adapter_rank,
        shared_depth=args.shared_depth,
        freeze_base=True,
        noise_scale=args.noise_scale,
    )
    
    # 3. 可选: 额外注入 LoRA 到基座 (更强的适应能力)
    if args.use_lora:
        print("Injecting LoRA layers...")
        inject_lora(
            model.base,
            rank=args.lora_rank,
            target_modules=["qkv", "proj"],
        )
        count_lora_parameters(model.base)
    
    return model


def create_dataloader(args):
    """创建数据加载器"""
    transform = PairedTransform(
        image_size=args.image_size,
        is_training=True,
    )
    
    dataset = PairedLayeredDataset(
        root_dir=args.data_path,
        transform=transform,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader


def train_one_epoch(model, loader, transport, optimizer, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for step, (images, masks, labels) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用 transport 计算 loss
        loss_dict = transport.training_losses(
            model=lambda x, t, y: model(x, t, y, control=masks),
            x1=images,
            model_kwargs={"y": labels},
        )
        
        loss = loss_dict["loss"].mean()
        loss.backward()
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
        
        optimizer.step()
        total_loss += loss.item()
        
        if step % args.log_every == 0:
            print(f"Epoch {epoch} Step {step}: loss = {loss.item():.4f}")
    
    return total_loss / len(loader)


def main(args):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    model = create_model(args)
    model = model.to(device)
    
    # 打印参数统计
    param_info = count_parameters(model)
    print(f"Total params: {param_info['total']:,}")
    print(f"Trainable params: {param_info['trainable']:,} ({param_info['trainable_ratio']:.2f}%)")
    
    # 创建数据加载器
    loader = create_dataloader(args)
    
    # 创建 transport
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
    )
    
    # 优化器 (只优化可训练参数)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model, loader, transport, optimizer, device, epoch, args
        )
        scheduler.step()
        
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # 保存 ControlNet 权重
            torch.save(
                {k: v for k, v in model.state_dict().items() if 'base.' not in k or 'lora' in k},
                output_dir / "best_controlnet.pt"
            )
            
            # 如果使用了 LoRA，单独保存
            if args.use_lora:
                save_lora_weights(model.base, str(output_dir / "best_lora.pt"))
            
            print(f"Saved best model at epoch {epoch}")
        
        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {k: v for k, v in model.state_dict().items() if 'base.' not in k or 'lora' in k},
                output_dir / f"checkpoint_{epoch:04d}.pt"
            )
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    # 模型
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=["SiT-S/2", "SiT-B/2", "SiT-L/2", "SiT-XL/2"])
    parser.add_argument("--base-ckpt", type=str, default=None,
                        help="预训练基座权重路径")
    
    # ControlNet 配置
    parser.add_argument("--adapter-rank", type=int, default=32,
                        help="轻量级 Adapter 的秩")
    parser.add_argument("--shared-depth", type=int, default=4,
                        help="共享权重的层数")
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        help="训练时噪声注入强度")
    
    # LoRA 配置
    parser.add_argument("--use-lora", action="store_true",
                        help="是否使用 LoRA")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA 秩")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Transport
    parser.add_argument("--path-type", type=str, default="Linear")
    parser.add_argument("--prediction", type=str, default="velocity")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="outputs/small_sample")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
