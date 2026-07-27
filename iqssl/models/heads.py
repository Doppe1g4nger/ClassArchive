"""Projectors, predictors and decoders.

Heads are **method-specific by design**. Barlow Twins' decorrelation objective
improves monotonically with projector width (its published configuration uses
8192); VICReg needs room for its covariance term; SimCLR works best with a
narrow 128-d output. Forcing one width on all of them would handicap several
methods while *looking* like fairness.

The control variable of this benchmark is the encoder. Heads are part of the
method, and the fairness argument rests on the equal tuning budget instead —
see the fairness contract in the package README.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPHead(nn.Module):
    """N-layer MLP with BatchNorm + ReLU between layers, linear output.

    BatchNorm is kept even when the encoder is a LayerNorm ViT. BYOL's and
    SimSiam's resistance to collapse is empirically tied to BN in the heads, so
    swapping it for LayerNorm silently changes what is being measured.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 3,
        *,
        norm: str = "bn",
        final_bn: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        def make_norm(d: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm1d(d)
            if norm == "ln":
                return nn.LayerNorm(d)
            if norm == "none":
                return nn.Identity()
            raise ValueError(f"unknown norm {norm!r}; options: bn, ln, none")

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(d, hidden_dim, bias=bias),
                make_norm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim, bias=bias))
        if final_bn:
            # Barlow Twins' reference implementation puts an affine-free BN on
            # the output; VICReg does not.
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def Projector(in_dim: int, hidden_dim: int = 2048, out_dim: int = 128, **kw) -> MLPHead:
    """3-layer projector, the SSL convention."""
    return MLPHead(in_dim, hidden_dim, out_dim, n_layers=3, **kw)


def Predictor(in_dim: int, hidden_dim: int = 512, out_dim: int | None = None, **kw) -> MLPHead:
    """2-layer predictor for the asymmetric methods.

    SimSiam's is a *bottleneck* (2048 -> 512 -> 2048); BYOL's is an expansion.
    The asymmetry is the collapse-avoidance mechanism, not an implementation
    detail — a predictor that is merely a wider linear map does not work.
    """
    return MLPHead(in_dim, hidden_dim, out_dim or in_dim, n_layers=2, **kw)


class LinearClassifier(nn.Module):
    """Frozen-feature probe: affine-free BatchNorm, then a linear layer.

    The BatchNorm is not decoration. It is MAE's own linear-probe protocol, and
    without it masked methods evaluate far worse than they should — their
    features have very different per-dimension scales than a contrastive
    method's. Omitting it would let the *protocol* decide the ranking.
    """

    def __init__(self, in_dim: int, n_classes: int, *, use_bn: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False) if use_bn else nn.Identity()
        self.fc = nn.Linear(in_dim, n_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.bn(x))


class PatchDecoder(nn.Module):
    """MAE decoder: a narrow transformer that reconstructs raw patches.

    Deliberately much smaller than the encoder (depth 4, dim 192). The decoder
    is discarded after pretraining; making it large would shift capacity into a
    component no downstream task ever uses.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        patch_size: int,
        in_ch: int = 2,
        decoder_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        from iqssl.models.vit1d import Block, sincos_positional_embedding

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "pos_embed", sincos_positional_embedding(num_patches, decoder_dim), persistent=False
        )
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * in_ch)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, latent: Tensor, ids_restore: Tensor) -> Tensor:
        """``latent`` is the encoder output over kept tokens (cls first).

        Mask tokens are appended, the sequence is un-shuffled back to input
        order, and every position is predicted.
        """
        x = self.decoder_embed(latent)
        cls, kept = x[:, :1], x[:, 1:]
        n_total = ids_restore.shape[1]
        pad = self.mask_token.expand(x.shape[0], n_total - kept.shape[1], -1)
        full = torch.cat([kept, pad], dim=1)
        full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        full = full + self.pos_embed  # type: ignore[operator]  # registered buffer
        x = torch.cat([cls, full], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))[:, 1:]
