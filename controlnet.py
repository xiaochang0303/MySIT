import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from timm.models.vision_transformer import PatchEmbed
except Exception as e:
    raise ImportError("ControlSiT requires timm PatchEmbed; please install timm.") from e

from models import SiT

class Zero(nn.Module):
    """
    一个以零初始化的线性层，常用于 ControlNet 结构中以防止训练初期对基座模型造成剧烈干扰。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)

class ControlAdapter(nn.Module):
    """
    控制适配器，通过多个块处理控制信息并生成残差。
    """
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        
        blocks, outs = [], []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ))
            outs.append(Zero(hidden_size))
            
        self.blocks = nn.ModuleList(blocks)
        self.outs = nn.ModuleList(outs)
        
        # 初始化
        for b in self.blocks:
            nn.init.zeros_(b[-1].weight)
            nn.init.zeros_(b[-1].bias)

    def forward(self, tokens: torch.Tensor):
        h = self.pre(tokens)  # 预处理
        residuals = []
        for block, proj in zip(self.blocks, self.outs):
            h = h + block(h)  # 残差连接的MLP处理
            residuals.append(proj(h))  # 通过零初始化投影层输出
        return residuals

def _infer_img_size(base: SiT) -> Tuple[int, int]:
    """
    从基座模型中推断图像尺寸的辅助函数。
    """
    if hasattr(base.x_embedder, 'img_size') and base.x_embedder.img_size is not None:
        sz = base.x_embedder.img_size
        if isinstance(sz, (list, tuple)):
            if len(sz) == 2:
                return int(sz[0]), int(sz[1])
            else:
                return int(sz[0]), int(sz[0])
        else:
            v = int(sz)
            return v, v
            
    assert hasattr(base.x_embedder, 'num_patches'), "x_embedder.num_patches not found"
    assert hasattr(base, 'patch_size'), "base.patch_size not found"
    grid = int(base.x_embedder.num_patches ** 0.5)
    H = W = grid * int(base.patch_size)
    return H, W

class ControlSiT(nn.Module):
    def __init__(self, base: SiT, freeze_base: bool = True, cfg_channels: str = "first3"):
        """
        Args:
            base: 预训练的 SiT 模型
            freeze_base: 是否冻结基座模型
            cfg_channels: CFG 应用的通道模式
                - "first3": 仅对前3通道应用 CFG (与原始 SiT 一致，用于精确复现)
                - "all": 对所有潜在通道应用 CFG (标准做法)
        """
        super().__init__()
        self.base = base
        self.cfg_channels = cfg_channels
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
                
        hidden = int(self.base.pos_embed.shape[-1])
        in_chans = int(self.base.in_channels)
        patch = int(self.base.patch_size)
        H, W = _infer_img_size(self.base)
        
        # Control-side PatchEmbed (镜像基座模型)
        self.control_embed = PatchEmbed(img_size=(H, W), patch_size=patch, in_chans=in_chans, embed_dim=hidden, bias=True)
        
        depth = len(self.base.blocks)
        self.adapter = ControlAdapter(hidden_size=hidden, depth=depth)

    @property
    def learn_sigma(self):
        return self.base.learn_sigma
        
    @property
    def in_channels(self):
        return self.base.in_channels

    @torch.no_grad()
    def _cond(self, t, y, training: bool):
        t_emb = self.base.t_embedder(t)
        y_emb = self.base.y_embedder(y, training)
        return t_emb + y_emb

    def forward(self, x, t, y, control: Optional[torch.Tensor] = None, control_strength: float = 1.0):
        x_tokens = self.base.x_embedder(x) + self.base.pos_embed
        c = self._cond(t, y, self.training)

        if control is not None:
            ctrl_tokens = self.control_embed(control) + self.base.pos_embed
            residuals = self.adapter(ctrl_tokens)
            # 应用控制强度
            residuals = [r * control_strength for r in residuals]
        else:
            residuals = [0.0] * len(self.base.blocks)

        h = x_tokens
        for i, block in enumerate(self.base.blocks):
            r = residuals[i]
            # 如果是float(即0.0)，则不加；否则加上残差
            h = block(h if isinstance(r, float) else (h + r), c)

        out = self.base.final_layer(h, c)
        out = self.base.unpatchify(out)

        if self.base.learn_sigma:
            out, _ = out.chunk(2, dim=1)
        return out

    def forward_with_cfg(self, x, t, y, cfg_scale, control: Optional[torch.Tensor] = None, control_strength: float = 1.0):
        # 只计算一半的数据，然后做CFG组合
        half = x[: len(x) // 2]  # (N, 4, H, W)
        combined = torch.cat([half, half], dim=0)  # (2N, 4, H, W)

        # 处理控制信号的维度对齐
        control_combined = None
        if control is not None:
            if control.shape[0] == x.shape[0]:  # 已经是 2N
                control_combined = control
            else:
                # 视为 (N, ...)，复制成 (2N, ...)
                control_half = control[: len(x) // 2]
                control_combined = torch.cat([control_half, control_half], dim=0)

        model_out = self.forward(combined, t, y, control=control_combined, control_strength=control_strength)

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