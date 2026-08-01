"""Patch-tokenizing 1D Vision Transformer — the shared default encoder.

Chosen as the common backbone because the masked and latent-prediction methods
(MAE, the three JEPAs) need patch tokens, and a fair comparison needs *one*
encoder rather than one per method family.

Two details are deliberate and load-bearing:

**Positional embeddings are fixed sinusoids, not learned.** Every EMA-teacher
method (BYOL, data2vec, both JEPAs) keeps a second copy of the encoder whose
weights lag the student's. If the positional table were learnable it would lag
too, so student and teacher would disagree about *where* each token is — an
asymmetry that has nothing to do with the objective under study.

**Three forward paths, not one.** MAE drops masked tokens entirely (the encoder
never sees them, which is where its speed comes from); data2vec replaces them
in place with a learned mask token, keeping sequence length; I-JEPA needs raw
token access at arbitrary index sets. Emulating any of these with the others
changes the method.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from iqssl.registry import ENCODERS


@dataclass
class EncoderOut:
    """Everything a method might want from one forward pass."""

    mean: Tensor
    """(B, D) mean over patch tokens. The pooling masked methods prefer."""
    cls: Tensor | None = None
    """(B, D) class token, when the encoder has one."""
    tokens: Tensor | None = None
    """(B, N, D) patch tokens, excluding the class token."""
    layers: list[Tensor] | None = None
    """Per-block token outputs. data2vec regresses their top-K average."""

    def pooled(self, how: str = "cls") -> Tensor:
        if how == "cls":
            if self.cls is None:
                raise ValueError("encoder has no cls token; use pool='mean'")
            return self.cls
        if how == "mean":
            return self.mean
        raise ValueError(f"unknown pooling {how!r}; options: cls, mean")


def sincos_positional_embedding(n: int, dim: int, device=None) -> Tensor:
    """Standard fixed sinusoidal table, ``(1, n, dim)``."""
    if dim % 2:
        raise ValueError(f"embedding dim must be even, got {dim}")
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    omega = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * omega)
    pe[:, 1::2] = torch.cos(pos * omega)
    return pe.unsqueeze(0)


class DropPath(nn.Module):
    """Stochastic depth on the residual branch."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(
        self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.drop_path(self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0])
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchStem(nn.Module):
    """Nonlinear tokenizer: one small CNN applied *within* each patch.

    The linear alternative (``nn.Conv1d(in_ch, dim, P, stride=P)``) is what a ViT
    normally uses, and on this data it does not work. The emitter fingerprint is
    second-order -- IQ imbalance lives in ``E[z²]``, PA compression in envelope
    variance, phase noise in ``dphi`` variance -- and a linear map of 16 raw
    samples cannot form any of them. The difficulty gate had already measured the
    same wall from the other side: ``SmallCNN`` scored 0.77 with mean pooling and
    0.891 once std pooling was added, because "an average cannot represent a
    second moment". Measured here, a linear-stem ViT sits at chance for its whole
    budget while ``cnn1d_tiny`` -- convolutions at full resolution, mean+std
    pooling -- learns immediately.

    So this mirrors what was measured to work: strided convs with a nonlinearity,
    then mean **and** std pooled into the token.

    Every conv is confined to a single patch, which is the constraint that makes
    this safe for the masked methods. A stem run across the whole sequence would
    give each token a receptive field several patches wide -- at kernel 7 and
    four stride-2 layers, 91 samples against a patch of 16 -- so MAE's kept
    tokens would already contain the content it is asked to reconstruct, and its
    "the encoder genuinely never sees 75% of the input" premise would be false
    while every test still passed. Reshaping the patch axis into the batch axis
    keeps the receptive field exactly one patch wide by construction.
    """

    def __init__(self, in_ch: int, patch_size: int, embed_dim: int, depth: int = 2) -> None:
        super().__init__()
        if patch_size % (2**depth):
            raise ValueError(
                f"patch_size {patch_size} must be divisible by {2**depth} for a "
                f"depth-{depth} stem; each layer halves the within-patch length"
            )
        self.patch_size = patch_size
        self.in_ch = in_ch

        width = max(embed_dim // 2, 16)
        chans = [in_ch] + [width] * depth
        layers: list[nn.Module] = []
        for a, b in itertools.pairwise(chans):
            layers += [
                nn.Conv1d(a, b, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(b),
                nn.ReLU(inplace=True),
            ]
        self.convs = nn.Sequential(*layers)
        # mean ++ std, hence 2x.
        self.proj = nn.Linear(width * 2, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, L)`` -> ``(B, N, D)``, N = L / patch_size."""
        b, c, length = x.shape
        n = length // self.patch_size
        # (B, C, L) -> (B*N, C, P): the patch axis becomes batch, so no
        # convolution can reach across a patch boundary.
        patches = x.reshape(b, c, n, self.patch_size).permute(0, 2, 1, 3).reshape(b * n, c, -1)

        h = self.convs(patches)
        # Biased std, and via the variance so the gradient at zero is finite --
        # a patch of identical samples is rare but not impossible (a dead
        # receiver, a zero-padded tail) and would otherwise produce NaNs.
        stats = torch.cat([h.mean(-1), h.var(-1, unbiased=False).clamp_min(1e-12).sqrt()], dim=-1)
        return self.proj(stats).view(b, n, -1)


class ViT1D(nn.Module):
    """1D ViT over IQ patches.

    Input ``(B, 2, L)`` float32; ``L / patch_size`` tokens. At the defaults
    (L=1024, patch=16) that is 64 tokens.

    ``stem`` selects how a patch becomes a token: ``"linear"`` is the textbook
    strided convolution, ``"conv"`` is :class:`PatchStem`. The choice is not
    cosmetic on this data -- see that class for the measurements.
    """

    supports_masking = True

    def __init__(
        self,
        in_ch: int = 2,
        seq_len: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        use_cls_token: bool = True,
        pool: str = "cls",
        stem: str = "linear",
    ) -> None:
        super().__init__()
        if seq_len % patch_size:
            raise ValueError(f"seq_len {seq_len} must be divisible by patch_size {patch_size}")
        if stem not in ("linear", "conv"):
            raise ValueError(f"unknown stem {stem!r}; options: linear, conv")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = seq_len // patch_size
        self.in_ch = in_ch
        self.pool = pool
        self.use_cls_token = use_cls_token

        self.stem_kind = stem
        self.patch_embed: nn.Module = (
            PatchStem(in_ch, patch_size, embed_dim)
            if stem == "conv"
            else nn.Conv1d(in_ch, embed_dim, patch_size, stride=patch_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        """Used by in-place masking (data2vec). MAE drops tokens instead."""

        self.register_buffer(
            "pos_embed",
            sincos_positional_embedding(self.num_patches, embed_dim),
            persistent=False,
        )

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # -- token construction ---------------------------------------------------

    def tokenize(self, x: Tensor) -> Tensor:
        """``(B, 2, L)`` -> ``(B, N, D)`` patch tokens with positions added."""
        # PatchStem already emits (B, N, D); the strided conv emits (B, D, N).
        h = self.patch_embed(x)
        return (h if self.stem_kind == "conv" else h.transpose(1, 2)) + self.pos_embed

    def _prepend_cls(self, tokens: Tensor) -> Tensor:
        if self.cls_token is None:
            return tokens
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)

    def _run_blocks(self, t: Tensor, collect: bool) -> tuple[Tensor, list[Tensor] | None]:
        layers: list[Tensor] | None = [] if collect else None
        for blk in self.blocks:
            t = blk(t)
            if layers is not None:
                layers.append(t)
        return t, layers

    # -- forward paths --------------------------------------------------------

    def forward(
        self, x: Tensor, *, return_tokens: bool = False, return_all_layers: bool = False
    ) -> EncoderOut:
        t = self._prepend_cls(self.tokenize(x))
        t, layers = self._run_blocks(t, return_all_layers)
        t = self.norm(t)

        offset = 1 if self.cls_token is not None else 0
        patches = t[:, offset:]
        return EncoderOut(
            mean=patches.mean(1),
            cls=t[:, 0] if self.cls_token is not None else None,
            tokens=patches if return_tokens else None,
            layers=[lyr[:, offset:] for lyr in layers] if layers else None,
        )

    def forward_masked(self, x: Tensor, keep_idx: Tensor) -> Tensor:
        """MAE path: encode **only** the kept tokens.

        ``keep_idx`` is ``(B, N_keep)`` int64. Returns ``(B, N_keep [+1], D)``
        including the class token when present. Dropping rather than masking is
        what makes MAE cheap — the encoder genuinely never sees 75% of the input.
        """
        tokens = self.tokenize(x)
        idx = keep_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        kept = torch.gather(tokens, 1, idx)
        t, _ = self._run_blocks(self._prepend_cls(kept), False)
        return self.norm(t)

    def forward_with_mask_token(self, x: Tensor, mask: Tensor) -> Tensor:
        """data2vec path: replace masked positions in place, keeping length.

        ``mask`` is ``(B, N)`` bool, True where masked. Returns ``(B, N, D)``.
        Sequence length is preserved so masked positions still attend and can be
        regressed at their original index.
        """
        tokens = self.tokenize(x)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        t, _ = self._run_blocks(self._prepend_cls(tokens), False)
        t = self.norm(t)
        return t[:, 1:] if self.cls_token is not None else t

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        """Run the blocks over pre-built tokens. The I-JEPA predictor path."""
        t, _ = self._run_blocks(tokens, False)
        return self.norm(t)


VIT_SIZES: dict[str, dict[str, int | float]] = {
    "tiny": {"embed_dim": 192, "depth": 6, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 8, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}


@ENCODERS.register("vit1d")
def vit1d(size: str = "small", **kwargs) -> ViT1D:
    """Named size shortcut. ``tiny`` is the CPU smoke-test configuration.

    The *factory* carries the registry entry, not the class: config-driven
    construction addresses encoders by size name, and registering the class would
    force every config to spell out embed_dim/depth/num_heads and let two
    experiments drift apart on a dimension the fairness contract holds fixed.
    """
    if size not in VIT_SIZES:
        raise ValueError(f"unknown ViT size {size!r}; options: {sorted(VIT_SIZES)}")
    return ViT1D(**{**VIT_SIZES[size], **kwargs})  # type: ignore[arg-type]
