"""1D residual CNN encoder — the secondary backbone.

Present as a sweep axis: "does the ranking of SSL objectives depend on the
architecture?" is a question the thesis should be able to answer rather than
assume.

**It deliberately does not support the masked methods.** MAE, both JEPAs and
data2vec all require patch tokens that can be dropped or replaced individually;
masked convolutions are a different research problem, and faking token dropping
by zeroing the input would silently change every one of those objectives. The
config validator refuses ``method=mae encoder=cnn1d`` with an explicit message
rather than letting it run and produce a number nobody should trust.
"""

from __future__ import annotations

from torch import Tensor, nn

from iqssl.models.vit1d import EncoderOut
from iqssl.registry import ENCODERS


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)
            )
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(out)) + identity)


class ResNet1D(nn.Module):
    """Residual 1D CNN over ``(B, 2, L)`` IQ, global-average pooled."""

    supports_masking = False

    def __init__(
        self,
        in_ch: int = 2,
        widths: tuple[int, ...] = (64, 128, 256, 512, 512),
        blocks: tuple[int, ...] = (2, 2, 2, 2, 2),
        stem_kernel: int = 15,
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks):
            raise ValueError(f"widths ({len(widths)}) and blocks ({len(blocks)}) must align")

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_ch, widths[0], stem_kernel, stride=2, padding=stem_kernel // 2, bias=False
            ),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
        )
        stages: list[nn.Module] = []
        c = widths[0]
        for w, n in zip(widths, blocks, strict=True):
            for i in range(n):
                stages.append(BasicBlock1D(c, w, stride=2 if i == 0 else 1))
                c = w
        self.stages = nn.Sequential(*stages)
        self.embed_dim = widths[-1]
        self.num_patches = 0  # no token grid; masked methods are rejected upstream

    def forward(
        self, x: Tensor, *, return_tokens: bool = False, return_all_layers: bool = False
    ) -> EncoderOut:
        h = self.stages(self.stem(x))  # (B, D, T)
        pooled = h.mean(-1)
        return EncoderOut(
            mean=pooled,
            cls=pooled,  # no class token; `pool='cls'` degrades to GAP
            tokens=h.transpose(1, 2) if return_tokens else None,
            layers=None,
        )

    def forward_masked(self, x: Tensor, keep_idx: Tensor) -> Tensor:
        raise NotImplementedError(
            "cnn1d does not support token dropping. Masked and latent-prediction "
            "methods (mae, ijepa, tsjepa, data2vec) require encoder=vit1d."
        )

    forward_with_mask_token = forward_masked


CNN_SIZES: dict[str, dict] = {
    "tiny": {"widths": (32, 64, 128, 128), "blocks": (1, 1, 1, 1)},
    "r18": {"widths": (64, 128, 256, 512, 512), "blocks": (2, 2, 2, 2, 2)},
}


@ENCODERS.register("cnn1d")
def cnn1d(size: str = "r18", seq_len: int | None = None, **kwargs) -> ResNet1D:
    """Named size shortcut.

    ``seq_len`` is accepted and ignored. ResNet1D is fully convolutional and
    global-average pools, so it has no fixed input length -- but the loop builds
    every encoder through one call signature, and special-casing which encoders
    take a length would put the branch back in exactly the place the design keeps
    it out of.
    """
    if size not in CNN_SIZES:
        raise ValueError(f"unknown CNN size {size!r}; options: {sorted(CNN_SIZES)}")
    return ResNet1D(**{**CNN_SIZES[size], **kwargs})
