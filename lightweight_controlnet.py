"""
轻量级 ControlNet 模块

针对小样本场景优化，相比原版 ControlNet 减少约 80% 参数量。
特点:
- 低秩投影替代全量 MLP
- 部分层共享权重
- 可学习的层级权重

Usage:
    from lightweight_controlnet import LightweightControlSiT
    from models import SiT_models
    
    # 加载预训练基座
    base = SiT_models["SiT-XL/2"](...)
    base.load_state_dict(torch.load("pretrained.pt"))
    
    # 创建轻量级 ControlNet
    model = LightweightControlSiT(base, rank=32, shared_depth=4)
    
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
    raise ImportError("LightweightControlSiT requires timm; please install timm.")

from models import SiT


class LowRankProjection(nn.Module):
    """
    低秩投影层: 用 down-up 结构替代大型 Linear
    
    参数量: 2 * hidden * rank, 而非 hidden * hidden
    当 rank << hidden 时，参数大幅减少
    """
    def __init__(self, hidden_size: int, rank: int = 32, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=bias)
        
        # 初始化: 使输出接近零
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        if bias:
            nn.init.zeros_(self.up.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SharedControlBlock(nn.Module):
    """
    共享权重的控制块
    
    多个层共享同一组 MLP 权重，但有独立的缩放因子
    大幅减少参数量
    """
    def __init__(self, hidden_size: int, rank: int = 32, num_layers: int = 4):
        super().__init__()
        
        # 共享的处理层
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            LowRankProjection(hidden_size, rank),
            nn.SiLU(),
        )
        
        # 每层独立的缩放 (很少的参数)
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_size))
            for _ in range(num_layers)
        ])
        
        # 每层独立的零初始化输出投影
        self.output_projs = nn.ModuleList([
            LowRankProjection(hidden_size, rank // 2)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of residuals, one per layer
        """
        shared_feat = self.shared_mlp(x)
        
        residuals = []
        for i in range(self.num_layers):
            # 缩放后的共享特征 + 独立投影
            scaled = shared_feat * (1 + self.layer_scales[i])
            residual = self.output_projs[i](scaled)
            residuals.append(residual)
            
        return residuals


class LightweightAdapter(nn.Module):
    """
    轻量级控制适配器
    
    结构:
    - 前几层使用共享权重 (SharedControlBlock)
    - 后几层使用独立的低秩投影
    - 整体参数量约为原版的 20%
    """
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        rank: int = 32,
        shared_depth: int = 4,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.shared_depth = min(shared_depth, depth)
        self.independent_depth = depth - self.shared_depth
        
        # 输入预处理
        self.pre = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            LowRankProjection(hidden_size, rank),
            nn.SiLU(),
        )
        
        # 共享权重块 (前 shared_depth 层)
        if self.shared_depth > 0:
            self.shared_block = SharedControlBlock(
                hidden_size, rank, self.shared_depth
            )
        else:
            self.shared_block = None
            
        # 独立层 (后 independent_depth 层)
        if self.independent_depth > 0:
            self.independent_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                    LowRankProjection(hidden_size, rank),
                    nn.SiLU(),
                    LowRankProjection(hidden_size, rank // 2),
                )
                for _ in range(self.independent_depth)
            ])
        else:
            self.independent_blocks = None
            
        # 可学习的全局层级权重 (控制每层残差的影响程度)
        self.layer_weights = nn.Parameter(torch.ones(depth) * 0.1)
        
    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            tokens: (B, N, C) 控制 token 序列
            
        Returns:
            List of (B, N, C) residuals, length = depth
        """
        h = self.pre(tokens)
        residuals = []
        
        # 共享块的残差
        if self.shared_block is not None:
            shared_residuals = self.shared_block(h)
            residuals.extend(shared_residuals)
            
        # 独立块的残差
        if self.independent_blocks is not None:
            for block in self.independent_blocks:
                out = block(h)
                h = h + out * 0.1  # 残差连接
                residuals.append(out)
                
        # 应用层级权重
        weights = torch.sigmoid(self.layer_weights)
        residuals = [r * w for r, w in zip(residuals, weights)]
        
        return residuals


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


class LightweightControlSiT(nn.Module):
    """
    轻量级 ControlSiT

    相比原版 ControlSiT 的改进:
    1. 使用低秩投影减少参数
    2. 部分层共享权重
    3. 可学习的层级权重
    4. 支持控制强度调节
    5. 训练时可选噪声注入增强多样性

    Args:
        base: 预训练的 SiT 模型
        rank: 低秩投影的秩
        shared_depth: 共享权重的层数
        freeze_base: 是否冻结基座
        noise_scale: 训练时噪声注入强度
        cfg_channels: CFG 应用的通道模式
            - "first3": 仅对前3通道应用 CFG (与原始 SiT 一致，用于精确复现)
            - "all": 对所有潜在通道应用 CFG (标准做法)
    """
    def __init__(
        self,
        base: SiT,
        rank: int = 32,
        shared_depth: int = 4,
        freeze_base: bool = True,
        noise_scale: float = 0.0,
        cfg_channels: str = "first3",
    ):
        super().__init__()
        self.base = base
        self.noise_scale = noise_scale
        self.cfg_channels = cfg_channels
        
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
        
        # 控制信号编码 (复用基座的 PatchEmbed 结构)
        self.control_embed = PatchEmbed(
            img_size=(H, W),
            patch_size=patch,
            in_chans=in_chans,
            embed_dim=hidden,
            bias=True,
        )
        
        # 轻量级适配器
        self.adapter = LightweightAdapter(
            hidden_size=hidden,
            depth=depth,
            rank=rank,
            shared_depth=shared_depth,
        )
        
        # 全局控制强度
        self.control_scale = nn.Parameter(torch.tensor(1.0))
        
        self._print_param_info()
        
    def _print_param_info(self):
        """打印参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_base = sum(p.numel() for p in self.base.parameters())
        ratio = trainable / total_base * 100
        print(f"[LightweightControlSiT] Trainable: {trainable:,} ({ratio:.2f}% of base)")
        
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
            # 对所有潜在通道应用 CFG (标准做法)
            cfg_ch = self.in_channels
        else:
            # 仅对前3通道应用 CFG (与原始 SiT 一致，用于精确复现)
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
