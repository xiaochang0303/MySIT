"""
LoRA (Low-Rank Adaptation) 模块

用于小样本数据集的高效微调，仅需训练约 0.1% 的参数量。
支持注入到 SiT/DiT 模型的 Attention 层。

Usage:
    from lora import inject_lora, save_lora_weights, load_lora_weights
    
    # 注入 LoRA 到模型
    lora_layers = inject_lora(model, rank=8, target_modules=["qkv", "proj"])
    
    # 只训练 LoRA 参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # 保存/加载 LoRA 权重
    save_lora_weights(model, "lora_dataset1.pt")
    load_lora_weights(model, "lora_dataset1.pt")
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import math


class LoRALayer(nn.Module):
    """
    低秩适配层: W' = W + BA, 其中 B ∈ R^{out×r}, A ∈ R^{r×in}
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: 低秩分解的秩，越小参数越少
        alpha: 缩放因子，默认等于 rank
        dropout: LoRA dropout 概率
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        
        # 低秩矩阵 A 和 B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化: A 用 Kaiming, B 用零初始化
        # 这样初始时 LoRA 输出为 0，不影响原模型
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # 是否启用 LoRA
        self.enabled = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                x.shape[0], x.shape[1], self.lora_B.out_features,
                device=x.device, dtype=x.dtype
            )
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
    
    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRALinear(nn.Module):
    """
    带 LoRA 的线性层包装器
    
    将原始 Linear 层包装，添加 LoRA 旁路
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # 冻结原始层
        for param in self.original.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)
    
    @property
    def weight(self):
        """兼容性: 返回原始权重"""
        return self.original.weight
    
    @property
    def bias(self):
        """兼容性: 返回原始偏置"""
        return self.original.bias


class LoRAQKV(nn.Module):
    """
    针对 timm VisionTransformer 的 QKV 层的 LoRA 包装
    
    timm 的 Attention 使用单个 Linear 同时生成 Q, K, V
    这个类为其添加 LoRA 适配
    """
    def __init__(
        self,
        original_qkv: nn.Linear,
        rank: int = 4,
        alpha: float = None,
        dropout: float = 0.0,
        enable_q: bool = True,
        enable_k: bool = False,
        enable_v: bool = True,
    ):
        super().__init__()
        self.original = original_qkv
        
        # 原始 QKV 输出维度 = 3 * hidden_size
        hidden_size = original_qkv.out_features // 3
        in_features = original_qkv.in_features
        
        # 为 Q, K, V 分别创建 LoRA (可选择性启用)
        self.lora_q = LoRALayer(in_features, hidden_size, rank, alpha, dropout) if enable_q else None
        self.lora_k = LoRALayer(in_features, hidden_size, rank, alpha, dropout) if enable_k else None
        self.lora_v = LoRALayer(in_features, hidden_size, rank, alpha, dropout) if enable_v else None
        
        self.hidden_size = hidden_size
        
        # 冻结原始层
        for param in self.original.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始 QKV 输出
        qkv = self.original(x)
        
        # 分离 Q, K, V
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        
        # 添加 LoRA 增量
        if self.lora_q is not None:
            q = q + self.lora_q(x)
        if self.lora_k is not None:
            k = k + self.lora_k(x)
        if self.lora_v is not None:
            v = v + self.lora_v(x)
            
        return torch.cat([q, k, v], dim=-1)


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = None,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    enable_qkv: Tuple[bool, bool, bool] = (True, False, True),
) -> Dict[str, nn.Module]:
    """
    向模型注入 LoRA 层
    
    Args:
        model: 目标模型 (SiT)
        rank: LoRA 秩
        alpha: 缩放因子
        dropout: Dropout 概率
        target_modules: 目标模块名称列表，默认 ["qkv", "proj"]
        enable_qkv: 对于 QKV 层，分别启用 Q, K, V 的 LoRA
        
    Returns:
        已注入的 LoRA 层字典
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]
    
    lora_layers = {}
    
    # 遍历模型中的所有 blocks
    for block_idx, block in enumerate(model.blocks):
        attn = block.attn
        
        # 处理 QKV 层
        if "qkv" in target_modules and hasattr(attn, "qkv"):
            original_qkv = attn.qkv
            lora_qkv = LoRAQKV(
                original_qkv,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                enable_q=enable_qkv[0],
                enable_k=enable_qkv[1],
                enable_v=enable_qkv[2],
            )
            attn.qkv = lora_qkv
            lora_layers[f"blocks.{block_idx}.attn.qkv"] = lora_qkv
            
        # 处理输出投影层
        if "proj" in target_modules and hasattr(attn, "proj"):
            original_proj = attn.proj
            lora_proj = LoRALinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            attn.proj = lora_proj
            lora_layers[f"blocks.{block_idx}.attn.proj"] = lora_proj
            
    print(f"[LoRA] Injected {len(lora_layers)} LoRA layers with rank={rank}")
    return lora_layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有 LoRA 参数"""
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """
    保存 LoRA 权重
    
    仅保存 LoRA 层的参数，文件很小（通常 < 10MB）
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRALinear, LoRAQKV)):
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    full_name = f"{name}.{param_name}"
                    lora_state_dict[full_name] = param.data.clone()
    
    torch.save(lora_state_dict, path)
    print(f"[LoRA] Saved {len(lora_state_dict)} parameters to {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """
    加载 LoRA 权重
    
    Args:
        model: 已注入 LoRA 的模型
        path: 权重文件路径
        strict: 是否严格匹配
    """
    lora_state_dict = torch.load(path, map_location="cpu")
    
    model_state = model.state_dict()
    loaded_keys = []
    
    for key, value in lora_state_dict.items():
        if key in model_state:
            model_state[key] = value
            loaded_keys.append(key)
        elif strict:
            raise KeyError(f"Key {key} not found in model")
            
    model.load_state_dict(model_state, strict=False)
    print(f"[LoRA] Loaded {len(loaded_keys)} parameters from {path}")


def set_lora_enabled(model: nn.Module, enabled: bool = True):
    """启用/禁用所有 LoRA 层"""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.enabled = enabled


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    统计 LoRA 参数量
    
    Returns:
        (lora_params, total_params, ratio)
    """
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    ratio = lora_params / total_params * 100
    
    print(f"[LoRA] Parameters: {lora_params:,} / {total_params:,} ({ratio:.2f}%)")
    return lora_params, total_params, ratio
