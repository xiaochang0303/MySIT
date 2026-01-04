"""
多样性采样模块

提供多种策略来增强生成图像的多样性，同时保持与控制信号的一致性。
适用于小样本数据增强场景。

Usage:
    from diverse_sampler import DiverseSampler, MaskAugmentor
    
    # 创建采样器
    sampler = DiverseSampler(model, transport, vae)
    
    # 单个 mask 生成多样化样本
    samples = sampler.sample_diverse(
        mask=mask_tensor,
        class_label=0,
        n_samples=10,
        diversity_config={
            "temperature_range": (0.8, 1.2),
            "cfg_range": (2.0, 6.0),
            "mask_augment": True,
        }
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass
import random


@dataclass
class DiversityConfig:
    """多样性采样配置"""
    # 温度范围
    temperature_min: float = 0.8
    temperature_max: float = 1.2
    
    # CFG scale 范围
    cfg_min: float = 2.0
    cfg_max: float = 6.0
    
    # Mask 增强
    mask_augment: bool = True
    mask_noise_std: float = 0.05
    mask_elastic_alpha: float = 5.0
    mask_elastic_sigma: float = 2.0
    
    # 控制强度变化
    control_strength_min: float = 0.8
    control_strength_max: float = 1.2
    
    # 采样步数变化
    num_steps_choices: List[int] = None
    
    def __post_init__(self):
        if self.num_steps_choices is None:
            self.num_steps_choices = [50, 100, 150]


class MaskAugmentor:
    """
    Mask 增强器
    
    提供多种 mask 变换方式来增加生成多样性
    """
    
    @staticmethod
    def add_noise(
        mask: torch.Tensor,
        noise_std: float = 0.05,
    ) -> torch.Tensor:
        """添加轻微噪声"""
        noise = torch.randn_like(mask) * noise_std
        return torch.clamp(mask + noise, 0, 1)
    
    @staticmethod
    def elastic_transform(
        mask: torch.Tensor,
        alpha: float = 5.0,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        弹性变形
        
        保持拓扑结构的同时产生形变
        """
        if mask.dim() == 4:
            B, C, H, W = mask.shape
        else:
            B, C, H, W = 1, 1, mask.shape[-2], mask.shape[-1]
            mask = mask.view(1, 1, H, W)
            
        device = mask.device
        
        # 生成随机位移场
        dx = torch.randn(B, 1, H, W, device=device) * alpha
        dy = torch.randn(B, 1, H, W, device=device) * alpha
        
        # 高斯平滑
        kernel_size = int(sigma * 6) | 1  # 确保奇数
        if kernel_size > 1:
            dx = _gaussian_blur(dx, kernel_size, sigma)
            dy = _gaussian_blur(dy, kernel_size, sigma)
        
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
        
        return warped
    
    @staticmethod
    def random_morphology(
        mask: torch.Tensor,
        kernel_size: int = 3,
        p_dilate: float = 0.3,
        p_erode: float = 0.3,
    ) -> torch.Tensor:
        """
        随机形态学操作 (膨胀/腐蚀)
        """
        if random.random() < p_dilate:
            mask = _dilate(mask, kernel_size)
        elif random.random() < p_erode:
            mask = _erode(mask, kernel_size)
        return mask
    
    @staticmethod
    def random_affine(
        mask: torch.Tensor,
        rotate_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_range: float = 0.05,
    ) -> torch.Tensor:
        """随机仿射变换"""
        angle = random.uniform(-rotate_range, rotate_range)
        scale = random.uniform(*scale_range)
        tx = random.uniform(-translate_range, translate_range)
        ty = random.uniform(-translate_range, translate_range)
        
        # 构建仿射矩阵
        theta = torch.tensor([
            [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), tx],
            [scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), ty],
        ], device=mask.device, dtype=mask.dtype).unsqueeze(0)
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
            
        grid = F.affine_grid(theta.expand(mask.shape[0], -1, -1), mask.shape, align_corners=True)
        warped = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return warped
    
    def augment(
        self,
        mask: torch.Tensor,
        config: DiversityConfig = None,
    ) -> torch.Tensor:
        """应用随机组合的增强"""
        if config is None:
            config = DiversityConfig()
            
        # 随机选择增强方式
        augmentations = []
        
        if random.random() < 0.5:
            augmentations.append(
                lambda m: self.add_noise(m, config.mask_noise_std)
            )
            
        if random.random() < 0.3:
            augmentations.append(
                lambda m: self.elastic_transform(m, config.mask_elastic_alpha, config.mask_elastic_sigma)
            )
            
        if random.random() < 0.3:
            augmentations.append(self.random_morphology)
            
        if random.random() < 0.3:
            augmentations.append(self.random_affine)
            
        # 应用增强
        result = mask
        for aug in augmentations:
            result = aug(result)
            
        return result


class DiverseSampler:
    """
    多样性采样器
    
    从单个或少量 mask 生成多样化的图像样本
    """
    
    def __init__(
        self,
        model: nn.Module,
        transport,  # Transport 对象
        vae: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.transport = transport
        self.vae = vae
        self.device = device
        self.mask_augmentor = MaskAugmentor()
        
    def _create_sampler(self, sampling_method: str, num_steps: int, **kwargs):
        """创建采样函数"""
        from transport import Sampler
        sampler = Sampler(self.transport)
        
        if sampling_method == "ode":
            return sampler.sample_ode(
                num_steps=num_steps,
                **kwargs
            )
        else:
            return sampler.sample_sde(
                num_steps=num_steps,
                **kwargs
            )
    
    def sample_single(
        self,
        mask: torch.Tensor,
        class_label: int,
        latent_size: int = 32,
        temperature: float = 1.0,
        cfg_scale: float = 4.0,
        control_strength: float = 1.0,
        num_steps: int = 50,
        sampling_method: str = "ode",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        生成单个样本
        
        Args:
            mask: (1, C, H, W) 控制 mask
            class_label: 类别标签
            latent_size: latent 空间大小
            temperature: 采样温度
            cfg_scale: CFG 强度
            control_strength: 控制强度
            num_steps: 采样步数
            sampling_method: "ode" 或 "sde"
            seed: 随机种子
            
        Returns:
            (1, 3, H, W) 生成的图像
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # 初始噪声
        z = torch.randn(1, 4, latent_size, latent_size, device=self.device) * temperature
        
        # 类别标签
        y = torch.tensor([class_label], device=self.device)
        
        # CFG 设置
        z_cfg = torch.cat([z, z], dim=0)
        y_null = torch.tensor([1000], device=self.device)  # null class
        y_cfg = torch.cat([y, y_null], dim=0)
        
        # Mask 处理
        if mask is not None:
            mask_cfg = torch.cat([mask, mask], dim=0)
        else:
            mask_cfg = None
        
        # 创建模型包装器
        def model_fn(x, t, **kwargs):
            return self.model.forward_with_cfg(
                x, t, y_cfg, cfg_scale,
                control=mask_cfg,
                control_strength=control_strength,
            )
        
        # 采样
        sample_fn = self._create_sampler(sampling_method, num_steps)
        samples = sample_fn(z_cfg, model_fn)[-1]
        samples = samples[:1]  # 只取条件生成的结果
        
        # VAE 解码
        if self.vae is not None:
            with torch.no_grad():
                samples = self.vae.decode(samples / 0.18215).sample
                
        return samples
    
    def sample_diverse(
        self,
        mask: torch.Tensor,
        class_label: int,
        n_samples: int,
        config: DiversityConfig = None,
        latent_size: int = 32,
        sampling_method: str = "ode",
        return_masks: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        生成多样化样本
        
        Args:
            mask: (1, C, H, W) 或 (C, H, W) 原始 mask
            class_label: 类别标签
            n_samples: 生成样本数
            config: 多样性配置
            latent_size: latent 空间大小
            sampling_method: 采样方法
            return_masks: 是否返回增强后的 masks
            
        Returns:
            生成的样本列表, 可选地返回对应的 masks
        """
        if config is None:
            config = DiversityConfig()
            
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(self.device)
        
        samples = []
        augmented_masks = []
        
        for i in range(n_samples):
            # 随机参数
            temperature = random.uniform(config.temperature_min, config.temperature_max)
            cfg_scale = random.uniform(config.cfg_min, config.cfg_max)
            control_strength = random.uniform(
                config.control_strength_min, config.control_strength_max
            )
            num_steps = random.choice(config.num_steps_choices)
            
            # Mask 增强
            if config.mask_augment:
                aug_mask = self.mask_augmentor.augment(mask.clone(), config)
            else:
                aug_mask = mask
                
            # 生成
            sample = self.sample_single(
                mask=aug_mask,
                class_label=class_label,
                latent_size=latent_size,
                temperature=temperature,
                cfg_scale=cfg_scale,
                control_strength=control_strength,
                num_steps=num_steps,
                sampling_method=sampling_method,
                seed=None,  # 每次随机
            )
            
            samples.append(sample)
            if return_masks:
                augmented_masks.append(aug_mask)
                
            print(f"Generated sample {i+1}/{n_samples} "
                  f"(temp={temperature:.2f}, cfg={cfg_scale:.2f}, ctrl={control_strength:.2f})")
        
        if return_masks:
            return samples, augmented_masks
        return samples
    
    def sample_interpolated(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        class_label: int,
        n_steps: int = 5,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        在两个 mask 之间插值生成
        
        生成从 mask1 到 mask2 的平滑过渡序列
        """
        samples = []
        alphas = torch.linspace(0, 1, n_steps)
        
        for alpha in alphas:
            interpolated_mask = mask1 * (1 - alpha) + mask2 * alpha
            sample = self.sample_single(
                mask=interpolated_mask,
                class_label=class_label,
                **kwargs,
            )
            samples.append(sample)
            
        return samples
    
    def sample_with_variations(
        self,
        mask: torch.Tensor,
        class_label: int,
        base_sample: torch.Tensor,
        n_variations: int = 5,
        variation_strength: float = 0.3,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        基于已有样本生成变体
        
        从一个基础生成结果出发，通过添加噪声生成相似但不同的变体
        """
        samples = []
        
        for i in range(n_variations):
            # 在基础样本上添加噪声，然后部分去噪
            noisy = base_sample + torch.randn_like(base_sample) * variation_strength
            
            # 使用较少的步数进行"修复"
            sample = self.sample_single(
                mask=mask,
                class_label=class_label,
                num_steps=20,  # 较少步数
                **kwargs,
            )
            samples.append(sample)
            
        return samples


# ============= 辅助函数 =============

def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """高斯模糊"""
    channels = x.shape[1]
    
    # 创建高斯核
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # 分离卷积
    kernel_h = g.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_v = g.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    
    padding = kernel_size // 2
    x = F.conv2d(x, kernel_h, padding=(0, padding), groups=channels)
    x = F.conv2d(x, kernel_v, padding=(padding, 0), groups=channels)
    
    return x


def _dilate(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """膨胀操作"""
    padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size, stride=1, padding=padding)


def _erode(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """腐蚀操作"""
    padding = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size, stride=1, padding=padding)


# ============= 便捷函数 =============

def generate_augmented_dataset(
    sampler: DiverseSampler,
    masks: List[torch.Tensor],
    labels: List[int],
    samples_per_mask: int = 10,
    config: DiversityConfig = None,
    save_dir: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    批量生成增强数据集
    
    Args:
        sampler: DiverseSampler 实例
        masks: mask 列表
        labels: 对应的标签列表
        samples_per_mask: 每个 mask 生成的样本数
        config: 多样性配置
        save_dir: 保存目录
        
    Returns:
        (生成的图像列表, 对应的 mask 列表, 对应的标签列表)
    """
    all_images = []
    all_masks = []
    all_labels = []
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        print(f"\nProcessing mask {i+1}/{len(masks)}")
        
        samples, aug_masks = sampler.sample_diverse(
            mask=mask,
            class_label=label,
            n_samples=samples_per_mask,
            config=config,
            return_masks=True,
        )
        
        all_images.extend(samples)
        all_masks.extend(aug_masks)
        all_labels.extend([label] * len(samples))
        
        # 可选保存
        if save_dir is not None:
            import os
            from torchvision.utils import save_image
            
            os.makedirs(save_dir, exist_ok=True)
            for j, (img, m) in enumerate(zip(samples, aug_masks)):
                save_image(img, f"{save_dir}/img_{i:04d}_{j:04d}.png", normalize=True)
                save_image(m, f"{save_dir}/mask_{i:04d}_{j:04d}.png")
                
    print(f"\nGenerated {len(all_images)} samples in total")
    return all_images, all_masks, all_labels
