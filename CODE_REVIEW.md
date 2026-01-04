# USDiT 项目代码审查报告

## 一、代码问题和需要修改的地方

### 1. 严重问题

#### 1.1 `train_small_sample.py:106-110` - Transport 调用方式错误

```python
loss_dict = transport.training_losses(
    model=lambda x, t, y: model(x, t, y, control=masks),
    x1=images,
    model_kwargs={"y": labels},
)
```

**问题**: `model_kwargs` 传递了 `y=labels`，但 lambda 函数已经接收了 `y` 参数。这会导致 `y` 被传递两次或者参数冲突。

**修复建议**:
```python
loss_dict = transport.training_losses(
    model=lambda x, t, **kwargs: model(x, t, kwargs.get('y'), control=masks),
    x1=images,
    model_kwargs={"y": labels},
)
```

---

#### 1.2 `lightweight_controlnet.py:182-183` - 独立块残差计算错误

```python
h = h + block(h) * 0.1  # 残差连接
residuals.append(block(h))  # 这里又调用了一次 block(h)
```

**问题**: `block(h)` 被调用了两次，第二次调用的输入是已经更新过的 `h`，这不是预期的行为，且浪费计算。

**修复建议**:
```python
block_out = block(h) * 0.1
h = h + block_out
residuals.append(block_out / 0.1)  # 或者直接 append block_out
```

---

#### 1.3 `diverse_sampler.py:302` - 采样函数返回值处理问题

```python
samples = sample_fn(z_cfg, model_fn)[-1]
```

**问题**: `sample_fn` 的返回值结构依赖于具体的采样方法，直接取 `[-1]` 可能不正确。ODE sampler 返回的是轨迹列表，但 SDE sampler 的返回格式可能不同。

---

### 2. 中等问题

#### 2.1 `train_controlnet.py:341` - 网格布局计算错误

```python
grid = make_grid(tiles, nrow=3*int(np.sqrt(n)), padding=2)
```

**问题**: 注释已经指出 `nrow` 应该是 3（每行显示：原图、边缘、生成图），但代码使用了 `3*int(np.sqrt(n))`。

**修复建议**:
```python
grid = make_grid(tiles, nrow=3, padding=2)
```

---

#### 2.2 `controlnet.py:131` 和 `lightweight_controlnet.py:336-337` - 残差添加位置不一致

```python
# controlnet.py
h = block(h if isinstance(r, float) else (h + r), c)

# lightweight_controlnet.py
if not isinstance(r, float):
    h = h + r
h = block(h, c)
```

**问题**: 两个 ControlNet 实现的残差添加逻辑不一致。`controlnet.py` 在 block 内部添加，`lightweight_controlnet.py` 在 block 外部添加。这可能导致行为差异。

---

#### 2.3 `models.py:261` 和 `controlnet.py:158` - CFG 只应用于前3通道

```python
eps, rest = model_out[:, :3], model_out[:, 3:]
```

**问题**: 硬编码只对前3通道应用 CFG，但 latent space 是4通道。注释说这是为了"exact reproducibility"，但这可能不是最优选择。

---

#### 2.4 `lora.py:273` - 使用已弃用的 `torch.load` 方式

```python
lora_state_dict = torch.load(path, map_location="cpu")
```

**建议**: 添加 `weights_only=True` 参数以提高安全性。

---

### 3. 轻微问题

#### 3.1 `maskdataset.py:89-90` - ImageWithCanny 中的 transform 应用顺序问题

```python
img_t = super().__call__(img)  # 这里调用了 transform
np_img = np.array(img)  # 但这里用的是原始 img
```

**问题**: Canny 边缘检测使用的是原始图像而不是 transform 后的图像，导致边缘和图像不对齐。

---

#### 3.2 `train_controlnet.py:381` - `is-training` 参数类型问题

```python
parser.add_argument("--is-training", type=bool, ...)
```

**问题**: `type=bool` 在 argparse 中不能正确解析 "True"/"False" 字符串。

**修复建议**:
```python
parser.add_argument("--is-training", action="store_true")
```

---

#### 3.3 缺少类型检查和输入验证

多个函数缺少对输入张量维度和类型的验证，可能导致难以调试的错误。

---

## 二、优化模型生成多样性的建议

### 1. 采样阶段优化

#### 1.1 增强 CFG 策略

当前实现使用固定的 CFG scale，建议：

- **动态 CFG**: 在采样过程中逐步调整 CFG scale（如从高到低）
- **CFG Rescale**: 参考 Imagen 论文，添加 CFG rescale 防止过饱和

```python
def cfg_rescale(cond_output, uncond_output, cfg_scale, rescale_factor=0.7):
    cfg_output = uncond_output + cfg_scale * (cond_output - uncond_output)
    # Rescale to prevent oversaturation
    std_cfg = cfg_output.std(dim=[1,2,3], keepdim=True)
    std_cond = cond_output.std(dim=[1,2,3], keepdim=True)
    rescaled = cfg_output * (std_cond / std_cfg) * rescale_factor + cfg_output * (1 - rescale_factor)
    return rescaled
```

#### 1.2 噪声调度优化

- **Truncated Sampling**: 不从纯噪声开始，而是从 t=0.8 或 0.9 开始
- **Noise Injection in SDE**: 增加 SDE 采样中的扩散系数以增加多样性

#### 1.3 温度采样

在 `diverse_sampler.py` 中已有温度参数，但可以进一步优化：

```python
# 在初始噪声中应用温度
z = torch.randn(...) * temperature

# 也可以在模型输出中应用温度
model_output = model_output / temperature
```

---

### 2. 训练阶段优化

#### 2.1 增强 Dropout 策略

当前 `class_dropout_prob=0.1`，建议：

- 增加 control signal dropout（随机丢弃控制信号）
- 添加 timestep-dependent dropout

```python
# 在 LightweightControlSiT.forward 中添加
if self.training and random.random() < self.control_dropout_prob:
    control = None  # 随机丢弃控制信号
```

#### 2.2 控制信号增强

`diverse_sampler.py` 中的 `MaskAugmentor` 很好，建议在训练时也使用：

- 弹性变形
- 随机形态学操作
- 添加噪声

#### 2.3 多尺度训练

- 在不同分辨率下训练
- 使用 progressive growing 策略

---

### 3. 模型架构优化

#### 3.1 控制强度的时间依赖性

当前控制强度是固定的，建议使其依赖于时间步：

```python
def forward(self, x, t, y, control, control_strength=1.0):
    # 在早期时间步（高噪声）减少控制强度
    # 在后期时间步（低噪声）增加控制强度
    t_normalized = t.view(-1, 1, 1)  # [0, 1]
    adaptive_strength = control_strength * (0.5 + 0.5 * t_normalized)
    ...
```

#### 3.2 多层控制注入策略

当前所有层使用相同的控制残差权重，建议：

```python
# 浅层：更多结构信息
# 深层：更多语义信息
layer_weights = torch.linspace(1.0, 0.5, depth)  # 或学习得到
```

#### 3.3 添加 Cross-Attention 机制

除了残差注入，可以添加 cross-attention 让模型更好地关注控制信号：

```python
class ControlCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x, control_tokens):
        # x: image tokens, control_tokens: control signal tokens
        return self.cross_attn(x, control_tokens, control_tokens)[0]
```

---

### 4. 数据增强优化

#### 4.1 Mixup/CutMix for Control Signals

```python
def mixup_control(ctrl1, ctrl2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    return lam * ctrl1 + (1 - lam) * ctrl2
```

#### 4.2 随机控制信号扰动

```python
def perturb_control(control, noise_level=0.1):
    noise = torch.randn_like(control) * noise_level
    return torch.clamp(control + noise, 0, 1)
```

---

### 5. 推理时多样性增强

#### 5.1 Ensemble Sampling

使用不同的随机种子和参数组合生成多个样本，然后选择最佳的：

```python
def ensemble_sample(model, control, n_candidates=5):
    candidates = []
    for i in range(n_candidates):
        seed = random.randint(0, 2**32)
        cfg = random.uniform(3.0, 6.0)
        sample = generate(model, control, seed=seed, cfg_scale=cfg)
        candidates.append(sample)
    # 可以用 CLIP score 或其他指标选择最佳
    return select_best(candidates)
```

#### 5.2 Latent Space Interpolation

```python
def interpolate_latents(z1, z2, alpha):
    # Spherical interpolation for better results
    theta = torch.acos(torch.sum(z1 * z2) / (z1.norm() * z2.norm()))
    return (torch.sin((1-alpha)*theta) * z1 + torch.sin(alpha*theta) * z2) / torch.sin(theta)
```

---

### 6. 具体实现建议

在 `diverse_sampler.py` 中添加以下功能：

```python
class EnhancedDiverseSampler(DiverseSampler):
    def sample_with_cfg_schedule(self, ...):
        """使用动态 CFG schedule 采样"""
        cfg_schedule = lambda t: cfg_max * (1 - t) + cfg_min * t
        ...

    def sample_with_noise_injection(self, ...):
        """在采样过程中注入额外噪声"""
        ...

    def sample_ensemble(self, ...):
        """集成多个采样结果"""
        ...
```

---

## 三、总结

### 优先级排序

#### 高优先级修复

| 问题 | 文件 | 行号 |
|------|------|------|
| Transport 调用方式错误 | `train_small_sample.py` | 106-110 |
| 独立块残差计算错误 | `lightweight_controlnet.py` | 182-183 |
| Canny 边缘对齐问题 | `maskdataset.py` | 89-90 |

#### 中优先级改进

| 问题 | 文件 | 行号 |
|------|------|------|
| 统一 ControlNet 残差注入逻辑 | `controlnet.py` / `lightweight_controlnet.py` | 131 / 336-337 |
| 网格布局计算错误 | `train_controlnet.py` | 341 |
| argparse bool 类型问题 | `train_controlnet.py` | 381 |

#### 多样性优化优先级

1. 实现动态 CFG 和 CFG rescale
2. 添加控制信号 dropout
3. 实现时间依赖的控制强度
4. 增强训练时的数据增强
