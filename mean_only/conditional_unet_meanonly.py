# conditional_unet_meanonly.py
# DDPM-style conditional U-Net (mean prediction only)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =======================================================
# 1. SINUSOIDAL TIME EMBEDDING
# =======================================================
def sinusoidal_time_embedding(timesteps, dim, max_period=10000):
    """
    Converts timestep t into a fixed positional encoding.
    Helps model learn denoising dynamics over time.

    Pros: non-learned, smooth interpolation across time.
    """
    half_dim = dim // 2
    exponents = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    frequencies = torch.exp(-math.log(max_period) * exponents)
    angles = timesteps[:, None].float() * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# =======================================================
# 2. CROSS-ATTENTION BLOCK
# =======================================================
class CrossAttentionBlock(nn.Module):
    """
    Injects external condition context via attention.

    Replaces plain additive 1x1 conv fusion.
    Learns what parts of the condition matter for x.
    """
    def __init__(self, channels, cond_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        # Safe GroupNorm selection
        def safe_groupnorm(num_channels):
            for g in reversed(range(1, 9)):  # Try groups from 8 to 1
                if num_channels % g == 0:
                    return nn.GroupNorm(g, num_channels)
            return nn.GroupNorm(1, num_channels)  # fallback

        self.norm_x = safe_groupnorm(channels)
        self.norm_cond = safe_groupnorm(cond_channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(cond_channels, channels, 1)
        self.v_proj = nn.Conv2d(cond_channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        _, _, Hc, Wc = cond.shape

        x = self.norm_x(x)
        cond = self.norm_cond(cond)

        q = self.q_proj(x).reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = self.k_proj(cond).reshape(B, self.num_heads, C // self.num_heads, Hc * Wc)
        v = self.v_proj(cond).reshape(B, self.num_heads, C // self.num_heads, Hc * Wc)

        attn = torch.einsum("bhcd,bhce->bhde", q, k) * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhde,bhce->bhcd", attn, v).reshape(B, C, H, W)
        return self.out_proj(out) + x


# =======================================================
# 3. SELF-ATTENTION BLOCK
# =======================================================
class SelfAttentionBlock(nn.Module):
    """
    Adds global spatial context to feature maps.
    Helps in large-scale structure learning (e.g., cloud fields).
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        def safe_groupnorm(num_channels):
            for g in reversed(range(1, 9)):
                if num_channels % g == 0:
                    return nn.GroupNorm(g, num_channels)
            return nn.GroupNorm(1, num_channels)

        self.norm = safe_groupnorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bhcd,bhce->bhde", q, k) * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhde,bhce->bhcd", attn, v).reshape(B, C, H, W)
        return self.proj(out) + x

# =======================================================
# 4. RESIDUAL BLOCK
# =======================================================
class ResidualBlock(nn.Module):
    """
    Backbone feature block: fuses time + cond + residual path.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.cross_attn = CrossAttentionBlock(out_channels, cond_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = self.conv1(self.act1(self.norm1(x)))
        t_proj = self.time_emb_proj(t_emb).view(x.shape[0], -1, 1, 1)

        if cond.shape[-2:] != h.shape[-2:]:
            cond = F.interpolate(cond, size=h.shape[-2:], mode='bilinear', align_corners=False)

        h = h + t_proj
        h = self.cross_attn(h, cond)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

# =======================================================
# 5. DOWNSAMPLE / UPSAMPLE
# =======================================================
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, in_channels, 4, 2, 1)
    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
    def forward(self, x):
        return self.op(x)
        
# =======================================================
# 6.1 MIDDLE BLOCK WRAPPER (to avoid nn.Sequential)
# =======================================================
class MiddleBlock(nn.Module):
    """
    Combines a ResidualBlock and SelfAttentionBlock,
    ensuring all required inputs are passed explicitly.
    """
    def __init__(self, channels, time_emb_dim, cond_channels):
        super().__init__()
        self.resblock = ResidualBlock(channels, channels, time_emb_dim, cond_channels)
        self.attn = SelfAttentionBlock(channels)

    def forward(self, x, t_emb, cond):
        x = self.resblock(x, t_emb, cond)
        return self.attn(x)

# =======================================================
# 6. CONDITIONAL DENOISING U-NET (MEAN ONLY)
# =======================================================
class ConditionalDenoisingUNet(nn.Module):
    """
    Final diffusion model:
      - Accepts noisy input and condition
      - Encodes timestep with sinusoidal embedding
      - Fuses information with cross + self attention
      - Predicts only mean of denoised sample

    Pros:
        ✓ Stable training
        ✓ Interpretable output
        ✓ Fewer parameters than mean+variance
    """
    def __init__(self, in_channels, model_channels, out_channels):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )

        # Encoder
        self.enc1 = ResidualBlock(1, model_channels, model_channels, in_channels)
        self.down1 = Downsample(model_channels)
        self.enc2 = ResidualBlock(model_channels, model_channels * 2, model_channels, in_channels)
        self.down2 = Downsample(model_channels * 2)
        self.enc3 = ResidualBlock(model_channels * 2, model_channels * 4, model_channels, in_channels)

        # Middle (w/ self-attention)
        # Middle (patched)
        self.middle = MiddleBlock(model_channels * 4, model_channels, in_channels)

        # Decoder
        self.dec3 = ResidualBlock(model_channels * 4, model_channels * 2, model_channels, in_channels)
        self.up2 = Upsample(model_channels * 2)
        self.dec2 = ResidualBlock(model_channels * 2, model_channels, model_channels, in_channels)
        self.up1 = Upsample(model_channels)
        self.dec1 = ResidualBlock(model_channels, model_channels, model_channels, in_channels)

        # Final output: only mean
        self.final = nn.Conv2d(model_channels, out_channels, kernel_size=1)

    def forward(self, x, t, cond):
        t_emb = sinusoidal_time_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)

        e1 = self.enc1(x, t_emb, cond)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb, cond)
        d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb, cond)

        m = self.middle(e3, t_emb, cond)

        u3 = self.dec3(m, t_emb, cond)
        u3 = F.interpolate(u3, size=e2.shape[-2:], mode='bilinear', align_corners=False) if u3.shape[-2:] != e2.shape[-2:] else u3
        u3 = u3 + e2
        u3 = self.up2(u3)

        u2 = self.dec2(u3, t_emb, cond)
        u2 = F.interpolate(u2, size=e1.shape[-2:], mode='bilinear', align_corners=False) if u2.shape[-2:] != e1.shape[-2:] else u2
        u2 = u2 + e1
        u1 = self.dec1(u2, t_emb, cond)

        return self.final(u1)
