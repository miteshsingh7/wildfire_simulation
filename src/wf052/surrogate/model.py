"""Wind-conditioned U-Net with Feature-wise Linear Modulation (FiLM) for fire arrival.

Contract:
- Inputs:
    - spatial_features: (B, 13, 256, 256)
        (7 fuel group channels + 6 continuous terrain/ignition channels)
    - wind_series: (B, 12, 2)
        (hourly wind u, v vectors)
- Outputs:
    - quantiles: (B, 3, 256, 256) -> [q10, q50, q90] arrival time in minutes
    - burn_logits: (B, 1, 256, 256) -> logit of P(burned by horizon)
- Parameter count: strictly 2-6M parameters.
- Compatible with AMP (torch.amp.autocast).
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ModelOutput(NamedTuple):
    """Container for surrogate model outputs."""

    quantiles: torch.Tensor  # (B, 3, H, W) -> [q10, q50, q90]
    burn_logits: torch.Tensor  # (B, 1, H, W) -> logit for P(burned)


class WindEncoder(nn.Module):
    """Encodes (B, 12, 2) hourly wind series into a dense embedding z_wind in R^emb_dim."""

    def __init__(self, in_features: int = 2, seq_len: int = 12, emb_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, wind_series: torch.Tensor) -> torch.Tensor:
        # wind_series: (B, 12, 2) -> transpose to (B, 2, 12) for 1D conv
        x = wind_series.transpose(1, 2)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return self.fc(x)  # (B, emb_dim)


class FiLMBlock(nn.Module):
    """Residual convolutional block modulated by FiLM (Feature-wise Linear Modulation).

    FiLM applies an affine transformation conditioned on wind embedding:
        FiLM(h) = (1 + gamma) * GroupNorm(h) + beta
    """

    def __init__(
        self, in_channels: int, out_channels: int, emb_dim: int = 64, num_groups: int = 8
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)

        self.film_proj = nn.Linear(emb_dim, out_channels * 2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

        # Initialize film projection so initial modulation is identity
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: torch.Tensor, z_wind: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM modulation: project z_wind -> (gamma, beta)
        film_params = self.film_proj(z_wind)  # (B, 2 * out_channels)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        h = (1.0 + gamma) * h + beta
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h + residual)
        return h


class WindConditionedUNet(nn.Module):
    """Wind-Conditioned U-Net for fire arrival time quantiles and burn probability.

    Total parameters are designed to be strictly in the 2-6M range.
    """

    def __init__(
        self,
        in_spatial_channels: int = 13,
        base_channels: int = 32,
        emb_dim: int = 64,
        max_horizon_min: float = 720.0,
    ) -> None:
        super().__init__()
        self.in_spatial_channels = in_spatial_channels
        self.max_horizon_min = max_horizon_min
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        # c1=32, c2=64, c3=128, c4=256

        self.wind_encoder = WindEncoder(in_features=2, seq_len=12, emb_dim=emb_dim)

        # Initial stem
        self.stem = nn.Conv2d(in_spatial_channels, c1, kernel_size=3, padding=1, bias=False)

        # Encoder stages
        self.enc1 = FiLMBlock(c1, c1, emb_dim=emb_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = FiLMBlock(c2, c2, emb_dim=emb_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = FiLMBlock(c3, c3, emb_dim=emb_dim)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottleneck (2 FiLM blocks)
        self.bottleneck1 = FiLMBlock(c4, c4, emb_dim=emb_dim)
        self.bottleneck2 = FiLMBlock(c4, c4, emb_dim=emb_dim)

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=False)
        self.dec3 = FiLMBlock(c4, c3, emb_dim=emb_dim)  # cat(c3, c3) = c4 -> c3

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.dec2 = FiLMBlock(c3, c2, emb_dim=emb_dim)  # cat(c2, c2) = c3 -> c2

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.dec1 = FiLMBlock(c2, c1, emb_dim=emb_dim)  # cat(c1, c1) = c2 -> c1

        # Prediction Heads
        # Head 1: Quantiles (q10, q50, q90) in minutes
        self.quantiles_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, 3, kernel_size=1),
        )

        # Head 2: Burn Logit (binary classification for unburned vs burned within horizon)
        self.burn_logit_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

    def init_burn_prob_bias(self, mean_burn_prob: float) -> float:
        """Initialise the bias of the burn-probability head to logit(clip(mean, 1e-4, 0.5)).

        Computed from training shards only.
        Returns the initialised scalar bias value.
        """
        clipped = min(max(float(mean_burn_prob), 1e-4), 0.5)
        bias_val = float(math.log(clipped / (1.0 - clipped)))
        with torch.no_grad():
            nn.init.constant_(self.burn_logit_head[2].bias, bias_val)
        return bias_val

    def forward(self, spatial_features: torch.Tensor, wind_series: torch.Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            spatial_features: (B, 13, H, W)
            wind_series: (B, 12, 2)

        Returns:
            ModelOutput with quantiles (B, 3, H, W) and burn_logits (B, 1, H, W).
        """
        z_wind = self.wind_encoder(wind_series)

        # Encoder
        x0 = self.stem(spatial_features)
        x1 = self.enc1(x0, z_wind)
        d1 = self.down1(x1)

        x2 = self.enc2(d1, z_wind)
        d2 = self.down2(x2)

        x3 = self.enc3(d2, z_wind)
        d3 = self.down3(x3)

        # Bottleneck
        b = self.bottleneck1(d3, z_wind)
        b = self.bottleneck2(b, z_wind)

        # Decoder
        u3 = self.up3(b)
        if u3.shape[2:] != x3.shape[2:]:
            u3 = F.interpolate(u3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        c3 = torch.cat([u3, x3], dim=1)
        y3 = self.dec3(c3, z_wind)

        u2 = self.up2(y3)
        if u2.shape[2:] != x2.shape[2:]:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        c2 = torch.cat([u2, x2], dim=1)
        y2 = self.dec2(c2, z_wind)

        u1 = self.up1(y2)
        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        c1 = torch.cat([u1, x1], dim=1)
        y1 = self.dec1(c1, z_wind)

        # Quantile predictions: bounded strictly in [0, max_horizon_min] using sigmoid
        raw_quantiles = self.quantiles_head(y1)
        quantiles = torch.sigmoid(raw_quantiles) * self.max_horizon_min

        # Burn logits: unconstrained logits
        burn_logits = self.burn_logit_head(y1)

        # Ensure output exactly matches input spatial dimensions if odd dimensions were rounded
        if quantiles.shape[2:] != spatial_features.shape[2:]:
            quantiles = F.interpolate(
                quantiles, size=spatial_features.shape[2:], mode="bilinear", align_corners=False
            )
            burn_logits = F.interpolate(
                burn_logits, size=spatial_features.shape[2:], mode="bilinear", align_corners=False
            )

        return ModelOutput(quantiles=quantiles, burn_logits=burn_logits)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
