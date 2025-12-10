"""
ControlNet for SiT
------------------
This module provides a ControlNet-style conditioning mechanism for SiT models.
It wraps a frozen SiT model and adds a trainable copy of the blocks (or a subset)
to inject control features.
"""
import torch
import torch.nn as nn
from models import SiT, SiTBlock, TimestepEmbedder, LabelEmbedder, PatchEmbed, modulate

class ControlNetSiT(nn.Module):
    """
    ControlNet wrapper for SiT.
    """
    def __init__(self, sit_model: SiT, control_channels: int = 3, hint_channels: int = 3):
        super().__init__()
        self.sit = sit_model
        
        # Freeze the main SiT model
        for param in self.sit.parameters():
            param.requires_grad = False
            
        self.hidden_size = sit_model.blocks[0].mlp.fc1.in_features # Get hidden size from first block
        self.patch_size = sit_model.patch_size
        self.in_channels = sit_model.in_channels
        
        # 1. Control Input Encoder (Hint Encoder)
        # Downsamples the control image (e.g. 256x256) to latent size (e.g. 32x32)
        # Assuming f=8 downsampling like VAE
        self.control_input_embedder = nn.Sequential(
            nn.Conv2d(hint_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), # 256->128
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, kernel_size=3, padding=1, stride=2), # 128->64
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, kernel_size=3, padding=1, stride=2), # 64->32 (Latent size)
            nn.SiLU(),
            nn.Conv2d(256, self.hidden_size, kernel_size=3, padding=1)
        )

        # 2. Control Blocks (Trainable copies of SiT blocks)
        # We use the same number of blocks as the main model
        self.control_blocks = nn.ModuleList([
            SiTBlock(self.hidden_size, sit_model.num_heads, mlp_ratio=4.0)
            for _ in range(len(sit_model.blocks))
        ])
        
        # Initialize control blocks with weights from main model
        for i, block in enumerate(self.control_blocks):
            block.load_state_dict(sit_model.blocks[i].state_dict())

        # 3. Zero Convs (Zero-initialized linear layers to inject features)
        # Since SiT works on sequences (N, T, D), we use Linear layers
        self.zero_convs = nn.ModuleList([
            self.make_zero_linear(self.hidden_size)
            for _ in range(len(sit_model.blocks))
        ])
        
        # Positional embedding for the control branch (reuse or new?)
        # We can reuse the main model's pos_embed since the grid size is the same
        
    def make_zero_linear(self, hidden_size):
        layer = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x, t, y, control):
        """
        x: (N, C, H, W) - input noisy latent
        t: (N,) - timestep
        y: (N,) - class label
        control: (N, Ch, H_img, W_img) - control image (e.g. edge map)
        """
        # 1. Prepare embeddings (Main Branch)
        x = self.sit.x_embedder(x) + self.sit.pos_embed  # (N, T, D)
        t_emb = self.sit.t_embedder(t)                   # (N, D)
        y_emb = self.sit.y_embedder(y, self.training)    # (N, D)
        c = t_emb + y_emb                                # (N, D)
        
        # 2. Prepare Control Features
        # Embed control image to latent feature map
        control_feat = self.control_input_embedder(control) # (N, D, H_lat, W_lat)
        
        # Flatten control features to match x: (N, D, H, W) -> (N, H*W, D)
        control_feat = control_feat.flatten(2).transpose(1, 2) # (N, T, D)
        
        # Add positional embedding to control features as well
        control_feat = control_feat + self.sit.pos_embed

        # 3. Joint Forward Pass
        # We pass the control features through the control blocks, 
        # and add the result to the main branch via zero convs.
        
        # Initial control state is the sum of x (noisy latent) and control features
        # This is a common design choice in ControlNet (adding to the input of the copy)
        control_x = x + control_feat
        
        for i, (main_block, control_block, zero_conv) in enumerate(zip(self.sit.blocks, self.control_blocks, self.zero_convs)):
            # Pass through Control Block
            control_x = control_block(control_x, c)
            
            # Pass through Zero Conv
            control_out = zero_conv(control_x)
            
            # Pass through Main Block and add Control Output
            # Note: In original ControlNet (U-Net), features are added to the decoder.
            # Here we have a flat transformer. We add to the input of the next block?
            # Or we add to the output of the current block?
            # Let's add to the output of the main block, effectively modifying the flow.
            
            main_out = main_block(x, c)
            x = main_out + control_out
            
        # Final Layer
        x = self.sit.final_layer(x, c)
        x = self.sit.unpatchify(x)
        
        if self.sit.learn_sigma:
            x, _ = x.chunk(2, dim=1)
            
        return x

    def forward_with_cfg(self, x, t, y, control, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        # For control, we usually duplicate it too
        control_combined = torch.cat([control, control], dim=0)
        
        model_out = self.forward(combined, t, y, control_combined)
        
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
