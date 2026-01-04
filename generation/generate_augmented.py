"""
使用多样性采样生成数据增强样本

展示如何从少量 mask 生成大量多样化训练数据

Usage:
    python generate_augmented.py \
        --model-ckpt outputs/small_sample/best_controlnet.pt \
        --base-ckpt pretrained.pt \
        --mask-dir /path/to/masks \
        --output-dir augmented_data \
        --samples-per-mask 20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

# 本地模块
from models import SiT_models, LightweightControlSiT
from models import inject_lora
from models.lora import load_lora_weights
from transport import create_transport
from generation.diverse_sampler import DiverseSampler, DiversityConfig, generate_augmented_dataset


def load_model(args, device):
    """加载模型"""
    # 基座
    base = SiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        learn_sigma=True,
    )
    
    if args.base_ckpt:
        base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    
    # 创建 ControlNet
    model = LightweightControlSiT(
        base=base,
        rank=args.adapter_rank,
        shared_depth=args.shared_depth,
        freeze_base=True,
    )
    
    # 加载 ControlNet 权重
    if args.model_ckpt:
        state_dict = torch.load(args.model_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded ControlNet weights from {args.model_ckpt}")
    
    # 可选: 加载 LoRA
    if args.lora_ckpt:
        inject_lora(model.base, rank=args.lora_rank)
        load_lora_weights(model.base, args.lora_ckpt)
        print(f"Loaded LoRA weights from {args.lora_ckpt}")
    
    model.eval()
    return model.to(device)


def load_masks(mask_dir, image_size):
    """加载 mask 文件"""
    mask_dir = Path(mask_dir)
    mask_paths = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    masks = []
    for path in mask_paths:
        img = Image.open(path).convert("L")  # 灰度
        mask = transform(img)
        masks.append(mask)
        
    print(f"Loaded {len(masks)} masks from {mask_dir}")
    return masks


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args, device)
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    
    # 创建 transport
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
    )
    
    # 创建采样器
    sampler = DiverseSampler(
        model=model,
        transport=transport,
        vae=vae,
        device=device,
    )
    
    # 配置多样性参数
    config = DiversityConfig(
        temperature_min=args.temp_min,
        temperature_max=args.temp_max,
        cfg_min=args.cfg_min,
        cfg_max=args.cfg_max,
        mask_augment=args.mask_augment,
        mask_noise_std=args.mask_noise_std,
        control_strength_min=args.ctrl_min,
        control_strength_max=args.ctrl_max,
        num_steps_choices=[args.num_steps],
    )
    
    # 加载 masks
    masks = load_masks(args.mask_dir, args.image_size)
    labels = [args.class_label] * len(masks)  # 统一标签
    
    # 生成增强数据
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_masks = []
    
    with torch.no_grad():
        for i, mask in enumerate(masks):
            print(f"\n{'='*50}")
            print(f"Processing mask {i+1}/{len(masks)}")
            print(f"{'='*50}")
            
            samples, aug_masks = sampler.sample_diverse(
                mask=mask,
                class_label=args.class_label,
                n_samples=args.samples_per_mask,
                config=config,
                latent_size=args.image_size // 8,
                sampling_method=args.sampling_method,
                return_masks=True,
            )
            
            # 保存
            for j, (img, m) in enumerate(zip(samples, aug_masks)):
                img_path = output_dir / "images" / f"{i:04d}_{j:04d}.png"
                mask_path = output_dir / "masks" / f"{i:04d}_{j:04d}.png"
                
                img_path.parent.mkdir(exist_ok=True)
                mask_path.parent.mkdir(exist_ok=True)
                
                save_image(img, img_path, normalize=True, value_range=(-1, 1))
                save_image(m, mask_path)
                
            all_images.extend(samples)
            all_masks.extend(aug_masks)
    
    print(f"\n{'='*50}")
    print(f"Generated {len(all_images)} images total")
    print(f"Saved to {output_dir}")
    print(f"{'='*50}")
    
    # 生成概览图
    if len(all_images) > 0:
        n_show = min(16, len(all_images))
        grid = torch.cat(all_images[:n_show], dim=0)
        save_image(
            grid, output_dir / "overview.png",
            nrow=4, normalize=True, value_range=(-1, 1)
        )
        print(f"Saved overview to {output_dir / 'overview.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 模型路径
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--model-ckpt", type=str, required=True,
                        help="ControlNet 权重路径")
    parser.add_argument("--lora-ckpt", type=str, default=None,
                        help="可选的 LoRA 权重路径")
    
    # 模型配置
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--shared-depth", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    # 数据路径
    parser.add_argument("--mask-dir", type=str, required=True,
                        help="Mask 文件目录")
    parser.add_argument("--output-dir", type=str, default="augmented_data")
    
    # 生成配置
    parser.add_argument("--samples-per-mask", type=int, default=10)
    parser.add_argument("--class-label", type=int, default=0)
    
    # 多样性参数
    parser.add_argument("--temp-min", type=float, default=0.8)
    parser.add_argument("--temp-max", type=float, default=1.2)
    parser.add_argument("--cfg-min", type=float, default=2.0)
    parser.add_argument("--cfg-max", type=float, default=6.0)
    parser.add_argument("--ctrl-min", type=float, default=0.8)
    parser.add_argument("--ctrl-max", type=float, default=1.2)
    parser.add_argument("--mask-augment", action="store_true", default=True)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    
    # 采样配置
    parser.add_argument("--sampling-method", type=str, default="ode",
                        choices=["ode", "sde"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--vae", type=str, default="mse",
                        choices=["ema", "mse"])
    
    # Transport
    parser.add_argument("--path-type", type=str, default="Linear")
    parser.add_argument("--prediction", type=str, default="velocity")
    
    args = parser.parse_args()
    main(args)
