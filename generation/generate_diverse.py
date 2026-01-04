"""
多样性生成脚本

从训练好的 ControlNet 模型生成多样化的图像，用于数据增广。

核心策略:
1. 温度采样 - 调整初始噪声的标准差
2. CFG 强度变化 - 不同的 classifier-free guidance 强度
3. 控制强度变化 - 调整 mask 对生成的影响程度
4. Mask 增强 - 对输入 mask 进行轻微变形
5. 多步采样 - 不同的采样步数

Usage:
    python generation/generate_diverse.py \
        --ckpt results/xxx/checkpoints/best.pt \
        --data-path sample_data \
        --output-dir augmented_data \
        --samples-per-image 10
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models import SiT_models, ControlSiT, LightweightControlSiT
from utils import find_model
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from training.train_utils import parse_transport_args
from datasets import PairedFlatDataset, PairedLayeredDataset, PairedTransform

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class DiversityConfig:
    """多样性生成配置"""
    # 温度范围 (初始噪声的标准差)
    temperature_min: float = 0.9
    temperature_max: float = 1.1

    # CFG 强度范围
    cfg_min: float = 3.0
    cfg_max: float = 6.0

    # 控制强度范围
    ctrl_strength_min: float = 0.7
    ctrl_strength_max: float = 1.3

    # Mask 增强
    mask_augment: bool = True
    mask_noise_std: float = 0.03
    mask_elastic_alpha: float = 3.0
    mask_elastic_sigma: float = 2.0

    # 采样步数选项
    num_steps_choices: List[int] = None

    def __post_init__(self):
        if self.num_steps_choices is None:
            self.num_steps_choices = [50]


class MaskAugmentor:
    """Mask 增强器"""

    @staticmethod
    def add_noise(mask: torch.Tensor, noise_std: float = 0.03) -> torch.Tensor:
        """添加轻微噪声"""
        noise = torch.randn_like(mask) * noise_std
        return torch.clamp(mask + noise, -1, 1)

    @staticmethod
    def elastic_transform(mask: torch.Tensor, alpha: float = 3.0, sigma: float = 2.0) -> torch.Tensor:
        """弹性变形"""
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        B, C, H, W = mask.shape
        device = mask.device

        # 生成随机位移场
        dx = torch.randn(B, 1, H, W, device=device) * alpha
        dy = torch.randn(B, 1, H, W, device=device) * alpha

        # 高斯平滑
        kernel_size = int(sigma * 6) | 1
        if kernel_size > 1:
            dx = MaskAugmentor._gaussian_blur(dx, kernel_size, sigma)
            dy = MaskAugmentor._gaussian_blur(dy, kernel_size, sigma)

        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # 添加位移
        dx_norm = dx.squeeze(1) / (W / 2)
        dy_norm = dy.squeeze(1) / (H / 2)
        offset = torch.stack([dx_norm, dy_norm], dim=-1)
        grid = grid + offset

        # 采样
        warped = F.grid_sample(mask, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped.squeeze(0) if warped.shape[0] == 1 else warped

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """高斯模糊"""
        channels = x.shape[1]
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        kernel_h = g.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        kernel_v = g.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

        padding = kernel_size // 2
        x = F.conv2d(x, kernel_h, padding=(0, padding), groups=channels)
        x = F.conv2d(x, kernel_v, padding=(padding, 0), groups=channels)
        return x

    @staticmethod
    def random_affine(mask: torch.Tensor, rotate_range: float = 5.0,
                      scale_range: Tuple[float, float] = (0.98, 1.02)) -> torch.Tensor:
        """随机仿射变换"""
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        angle = random.uniform(-rotate_range, rotate_range)
        scale = random.uniform(*scale_range)

        theta = torch.tensor([
            [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), 0],
            [scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), 0],
        ], device=mask.device, dtype=mask.dtype).unsqueeze(0)

        grid = F.affine_grid(theta.expand(mask.shape[0], -1, -1), mask.shape, align_corners=True)
        warped = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped.squeeze(0) if warped.shape[0] == 1 else warped

    def augment(self, mask: torch.Tensor, config: DiversityConfig) -> torch.Tensor:
        """应用随机组合的增强"""
        result = mask

        if random.random() < 0.5:
            result = self.add_noise(result, config.mask_noise_std)

        if random.random() < 0.3:
            result = self.elastic_transform(result, config.mask_elastic_alpha, config.mask_elastic_sigma)

        if random.random() < 0.3:
            result = self.random_affine(result)

        return result


def load_model(args, device):
    """加载模型"""
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # 加载 checkpoint
    checkpoint = find_model(args.ckpt)

    # 判断模型类型
    if args.use_lightweight:
        model = LightweightControlSiT(
            base,
            rank=args.light_rank,
            shared_depth=args.light_shared_depth,
            freeze_base=True,
        )
    else:
        model = ControlSiT(base, freeze_base=True)

    # 加载权重
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"], strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model


def generate_single(
    model, vae, sampler, mask_latent, class_label,
    latent_size, device, config: DiversityConfig,
    mask_augmentor: MaskAugmentor = None,
    num_classes: int = 1000,
):
    """生成单个样本"""
    # 随机参数
    temperature = random.uniform(config.temperature_min, config.temperature_max)
    cfg_scale = random.uniform(config.cfg_min, config.cfg_max)
    ctrl_strength = random.uniform(config.ctrl_strength_min, config.ctrl_strength_max)
    num_steps = random.choice(config.num_steps_choices)

    # Mask 增强
    if config.mask_augment and mask_augmentor is not None:
        aug_mask = mask_augmentor.augment(mask_latent.clone(), config)
    else:
        aug_mask = mask_latent

    # 初始噪声
    z = torch.randn(1, 4, latent_size, latent_size, device=device) * temperature

    # 类别标签
    y = torch.tensor([class_label], device=device)

    # CFG 设置
    z_in = torch.cat([z, z], dim=0)
    y_null = torch.tensor([num_classes], device=device)
    y_in = torch.cat([y, y_null], dim=0)

    # 采样
    sample_fn = sampler.sample_ode(num_steps=num_steps)

    with torch.no_grad():
        samples = sample_fn(
            z_in, model.forward_with_cfg,
            control=aug_mask, control_strength=ctrl_strength,
            y=y_in, cfg_scale=cfg_scale
        )[-1]

        # 取条件生成的结果
        samples, _ = samples.chunk(2, dim=0)

        # VAE 解码
        samples = vae.decode(samples / 0.18215).sample.clamp(-1, 1)

    return samples, aug_mask


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args, device)

    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()

    # Transport
    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight,
        args.train_eps, args.sample_eps
    )
    sampler = Sampler(transport)

    # 多样性配置
    config = DiversityConfig(
        temperature_min=args.temp_min,
        temperature_max=args.temp_max,
        cfg_min=args.cfg_min,
        cfg_max=args.cfg_max,
        ctrl_strength_min=args.ctrl_min,
        ctrl_strength_max=args.ctrl_max,
        mask_augment=args.mask_augment,
        mask_noise_std=args.mask_noise_std,
        num_steps_choices=[args.num_steps],
    )

    mask_augmentor = MaskAugmentor()
    latent_size = args.image_size // 8

    # 数据集
    transform = PairedTransform(args.image_size, is_training=False)

    if os.path.exists(os.path.join(args.data_path, 'images')):
        dataset = PairedFlatDataset(args.data_path, transform=transform, class_label=args.class_label)
    else:
        dataset = PairedLayeredDataset(args.data_path, transform=transform)

    print(f"Dataset contains {len(dataset)} images")

    # 输出目录
    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    # 生成
    total_generated = 0

    for idx in range(len(dataset)):
        (img, mask), label, filename = dataset[idx]

        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        # 编码 mask 到 latent
        with torch.no_grad():
            if mask.shape[1] == 1:
                mask_rgb = mask.repeat(1, 3, 1, 1)
            else:
                mask_rgb = mask
            mask_latent = vae.encode(mask_rgb).latent_dist.sample().mul_(0.18215)

        print(f"\nProcessing {idx+1}/{len(dataset)}: {filename}")

        for j in range(args.samples_per_image):
            # 生成
            gen_img, aug_mask_latent = generate_single(
                model, vae, sampler, mask_latent, label,
                latent_size, device, config, mask_augmentor,
                num_classes=args.num_classes
            )

            # 保存
            img_path = output_dir / "images" / f"{filename}_{j:03d}.png"
            save_image(gen_img, img_path, normalize=True, value_range=(-1, 1))

            # 解码并保存增强后的 mask
            with torch.no_grad():
                aug_mask_decoded = vae.decode(aug_mask_latent / 0.18215).sample.clamp(-1, 1)
            mask_path = output_dir / "masks" / f"{filename}_{j:03d}.png"
            save_image(aug_mask_decoded, mask_path, normalize=True, value_range=(-1, 1))

            total_generated += 1

            if (j + 1) % 5 == 0:
                print(f"  Generated {j+1}/{args.samples_per_image}")

    print(f"\nDone! Generated {total_generated} images")
    print(f"Saved to {output_dir}")

    # 生成概览图
    if total_generated > 0:
        print("Creating overview grid...")
        overview_images = []
        for img_path in sorted((output_dir / "images").glob("*.png"))[:16]:
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            overview_images.append(img_tensor)

        if overview_images:
            grid = torch.stack(overview_images)
            save_image(grid, output_dir / "overview.png", nrow=4, padding=2)
            print(f"Overview saved to {output_dir / 'overview.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 模型
    parser.add_argument("--ckpt", type=str, required=True, help="ControlNet checkpoint")
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="ema")

    # ControlNet 类型
    parser.add_argument("--use-lightweight", action="store_true")
    parser.add_argument("--light-rank", type=int, default=32)
    parser.add_argument("--light-shared-depth", type=int, default=4)

    # 数据
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="augmented_data")
    parser.add_argument("--class-label", type=int, default=0)

    # 生成配置
    parser.add_argument("--samples-per-image", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=50)

    # 多样性参数
    parser.add_argument("--temp-min", type=float, default=0.9)
    parser.add_argument("--temp-max", type=float, default=1.1)
    parser.add_argument("--cfg-min", type=float, default=3.0)
    parser.add_argument("--cfg-max", type=float, default=6.0)
    parser.add_argument("--ctrl-min", type=float, default=0.7)
    parser.add_argument("--ctrl-max", type=float, default=1.3)
    parser.add_argument("--mask-augment", action="store_true", default=True)
    parser.add_argument("--no-mask-augment", action="store_false", dest="mask_augment")
    parser.add_argument("--mask-noise-std", type=float, default=0.03)

    # Transport
    parse_transport_args(parser)

    args = parser.parse_args()
    main(args)
