"""Encoders and heads.

The encoder is the thing held **fixed** across every method — it is the control
variable of the whole benchmark. Projectors and predictors are not: Barlow Twins
and VICReg genuinely need wide projectors, and capping them at a contrastive
method's 128 dimensions would be a handicap dressed up as fairness.
"""

from iqssl.models.cnn1d import ResNet1D
from iqssl.models.ema import EMATeacher
from iqssl.models.heads import MLPHead, Predictor, Projector
from iqssl.models.vit1d import EncoderOut, ViT1D

__all__ = ["EMATeacher", "EncoderOut", "MLPHead", "Predictor", "Projector", "ResNet1D", "ViT1D"]
