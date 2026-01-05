"""
中等规模 ControlNet 模块

介于 ControlSiT (100%) 和 LightweightControlSiT (20%) 之间，
参数量约为原版的 40-50%。

设计思路:
- 使用 bottleneck MLP (hidden → hidden//2 → hidden) 替代全量 MLP
- 每层独立处理，不共享权重（保持表达能力）
- 分组残差注入：相邻层共享部分计算
- 可学习的层级权重

Usage:
    from models.medium_controlnet import MediumControlSiT
    from models import SiT_models

    # 加载预训练基座
    base = SiT_models["SiT-XL/2"](...)
    base.load_state_dict(torch.load("pretrained.pt"))

    # 创建中等规模 ControlNet
    model = MediumControlSiT(base, bottleneck_ratio=0.5)

    # 训练时只有 control 部分的参数可训练
    trainable_params = [p for p in model.parameters() if p.requires_grad]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

try:
    from timm.models.vision_transformer import PatchEmbed
except ImportError:
    raise ImportError("MediumControlSiT requires timm; please install timm.")

from .models import SiT


class ZeroLinear(nn.Module):
    """
    零初始化的线性层，用于 ControlNet 结构中防止训练初期对基座模型造成干扰。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP: hidden → bottleneck → hidden

    相比原版 ControlAdapter 的 hidden → 4*hidden → hidden，
    使用 hidden → hidden*ratio → hidden，减少约 50% 参数。
    """
    def __init__(
        self,
        hidden_size: int,
        bottleneck_ratio: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        bottleneck_dim = int(hidden_size * bottleneck_ratio)

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, bottleneck_dim, bias=bias),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_size, bias=bias),
        )

        # 初始化最后一层为零，保证训练初期不干扰基座
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MediumControlBlock(nn.Module):
    """
    中等规模控制块

    每个块处理一组相邻层（group_size 层），共享部分计算但保持独立输出。
    这样既减少参数，又保持足够的表达能力。
    """
    def __init__(
        self,
        hidden_size: int,
        group_size: int = 2,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()
        self.group_size = group_size
        bottleneck_dim = int(hidden_size * bottleneck_ratio)

        # 共享的特征提取
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, bottleneck_dim),
            nn.SiLU(),
        )

        # 每层独立的输出投影（零初始化）
        self.output_projs = nn.ModuleList([
            ZeroLinear(bottleneck_dim, hidden_size)
            for _ in range(group_size)
        ])

        # 每层独立的残差 MLP（较小）
        self.residual_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, bottleneck_dim // 2),
                nn.SiLU(),
                nn.Linear(bottleneck_dim // 2, hidden_size),
            )
            for _ in range(group_size)
        ])

        # 初始化残差 MLP 的最后一层
        for mlp in self.residual_mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, N, C) 输入 token

        Returns:
            List of (B, N, C) residuals, length = group_size
        """
        # 共享特征
        shared_feat = self.shared_proj(x)

        residuals = []
        h = x
        for i in range(self.group_size):
            # 共享特征 + 独立输出
            out = self.output_projs[i](shared_feat)
            # 加上独立的残差处理
            out = out + self.residual_mlps[i](h)
            residuals.append(out)
            # 更新 h 用于下一层
            h = h + out * 0.1

        return residuals


class MediumControlAdapter(nn.Module):
    """
    中等规模控制适配器

    结构:
    - 输入预处理层
    - 分组处理：每 group_size 层共享部分计算
    - 每层独立的输出投影
    - 可学习的层级权重

    参数量约为原版 ControlAdapter 的 40-50%
    """
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        bottleneck_ratio: float = 0.5,
        group_size: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        self.group_size = group_size
        self.num_groups = (depth + group_size - 1) // group_size

        # 输入预处理
        bottleneck_dim = int(hidden_size * bottleneck_ratio)
        self.pre = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_size),
        )

        # 分组控制块
        self.blocks = nn.ModuleList()
        remaining = depth
        for _ in range(self.num_groups):
            gs = min(group_size, remaining)
            self.blocks.append(
                MediumControlBlock(hidden_size, gs, bottleneck_ratio)
            )
            remaining -= gs

        # 可学习的层级权重
        self.layer_weights = nn.Parameter(torch.ones(depth) * 0.5)

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            tokens: (B, N, C) 控制 token 序列

        Returns:
            List of (B, N, C) residuals, length = depth
        """
        h = self.pre(tokens)

        all_residuals = []
        for block in self.blocks:
            residuals = block(h)
            all_residuals.extend(residuals)
            # 更新 h
            if residuals:
                h = h + residuals[-1] * 0.1

        # 截断到正确的深度
        all_residuals = all_residuals[:self.depth]

        # 应用层级权重
        weights = torch.sigmoid(self.layer_weights)
        all_residuals = [r * w for r, w in zip(all_residuals, weights)]

        return all_residuals


def _infer_img_size(base: SiT) -> Tuple[int, int]:
    """从基座模型推断图像尺寸"""
    if hasattr(base.x_embedder, 'img_size') and base.x_embedder.img_size is not None:
        sz = base.x_embedder.img_size
        if isinstance(sz, (list, tuple)):
            return (int(sz[0]), int(sz[1])) if len(sz) == 2 else (int(sz[0]), int(sz[0]))
        return int(sz), int(sz)

    grid = int(base.x_embedder.num_patches ** 0.5)
    H = W = grid * int(base.patch_size)
    return H, W


class MediumControlSiT(nn.Module):
    """
    中等规模 ControlSiT

    介于 ControlSiT 和 LightweightControlSiT 之间:
    - ControlSiT: 100% 参数量，每层独立 MLP (hidden → 4*hidden → hidden)
    - MediumControlSiT: 40-50% 参数量，bottleneck MLP + 分组处理
    - LightweightControlSiT: 20% 参数量，低秩投影 + 共享权重

    Args:
        base: 预训练的 SiT 模型
        bottleneck_ratio: bottleneck 维度比例，默认 0.5
        group_size: 分组大小，默认 2（每 2 层共享部分计算）
        freeze_base: 是否冻结基座
        noise_scale: 训练时噪声注入强度
        cfg_channels: CFG 应用的通道模式
            - "first3": 仅对前3通道应用 CFG
            - "all": 对所有潜在通道应用 CFG
    """
    def __init__(
        self,
        base: SiT,
        bottleneck_ratio: float = 0.5,
        group_size: int = 2,
        freeze_base: bool = True,
        noise_scale: float = 0.0,
        cfg_channels: str = "first3",
    ):
        super().__init__()
        self.base = base
        self.noise_scale = noise_scale
        self.cfg_channels = cfg_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.group_size = group_size

        # 冻结基座
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # 获取模型参数
        hidden = int(base.pos_embed.shape[-1])
        in_chans = int(base.in_channels)
        patch = int(base.patch_size)
        H, W = _infer_img_size(base)
        depth = len(base.blocks)

        # 控制信号编码
        self.control_embed = PatchEmbed(
            img_size=(H, W),
            patch_size=patch,
            in_chans=in_chans,
            embed_dim=hidden,
            bias=True,
        )

        # 中等规模适配器
        self.adapter = MediumControlAdapter(
            hidden_size=hidden,
            depth=depth,
            bottleneck_ratio=bottleneck_ratio,
            group_size=group_size,
        )

        # 全局控制强度
        self.control_scale = nn.Parameter(torch.tensor(1.0))

        self._print_param_info()

    def _print_param_info(self):
        """打印参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_base = sum(p.numel() for p in self.base.parameters())
        ratio = trainable / total_base * 100
        print(f"[MediumControlSiT] Trainable: {trainable:,} ({ratio:.2f}% of base)")
        print(f"  - bottleneck_ratio: {self.bottleneck_ratio}")
        print(f"  - group_size: {self.group_size}")

    @property
    def learn_sigma(self):
        return self.base.learn_sigma

    @property
    def in_channels(self):
        return self.base.in_channels

    def _cond(self, t, y, training: bool):
        """计算时间和类别条件嵌入"""
        with torch.no_grad():
            t_emb = self.base.t_embedder(t)
            y_emb = self.base.y_embedder(y, training)
        return t_emb + y_emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        control_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 噪声图像
            t: (B,) 时间步
            y: (B,) 类别标签
            control: (B, C, H, W) 控制信号 (如 mask)
            control_strength: 控制强度，0.0-2.0

        Returns:
            (B, C, H, W) 预测输出
        """
        # 图像 token 编码
        x_tokens = self.base.x_embedder(x) + self.base.pos_embed
        c = self._cond(t, y, self.training)

        # 控制信号处理
        if control is not None:
            ctrl_tokens = self.control_embed(control) + self.base.pos_embed

            # 训练时添加噪声增强多样性
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(ctrl_tokens) * self.noise_scale
                ctrl_tokens = ctrl_tokens + noise

            residuals = self.adapter(ctrl_tokens)

            # 应用控制强度
            scale = control_strength * self.control_scale
            residuals = [r * scale for r in residuals]
        else:
            residuals = [0.0] * len(self.base.blocks)

        # 通过基座 Transformer blocks
        h = x_tokens
        for i, block in enumerate(self.base.blocks):
            r = residuals[i]
            if not isinstance(r, float):
                h = h + r
            h = block(h, c)

        # 最终层
        out = self.base.final_layer(h, c)
        out = self.base.unpatchify(out)

        if self.base.learn_sigma:
            out, _ = out.chunk(2, dim=1)

        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        control: Optional[torch.Tensor] = None,
        control_strength: float = 1.0,
    ) -> torch.Tensor:
        """带 Classifier-Free Guidance 的前向传播"""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # 控制信号对齐
        ctrl_combined = None
        if control is not None:
            if control.shape[0] == x.shape[0]:
                ctrl_combined = control
            else:
                ctrl_half = control[: len(x) // 2]
                ctrl_combined = torch.cat([ctrl_half, ctrl_half], dim=0)

        model_out = self.forward(
            combined, t, y,
            control=ctrl_combined,
            control_strength=control_strength,
        )

        # 根据 cfg_channels 选择 CFG 应用的通道
        if self.cfg_channels == "all":
            cfg_ch = self.in_channels
        else:
            cfg_ch = 3

        eps, rest = model_out[:, :cfg_ch], model_out[:, cfg_ch:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / total * 100 if total > 0 else 0,
    }
