# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Feature alignment modules for knowledge distillation."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class ConvAligner(nn.Module):
    """1x1 Conv -> ReLU -> 1x1 Conv. Maps student features to teacher feature dimensions."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """Initialize ConvAligner.

        Args:
            in_channels (int): Input channels (student feature).
            out_channels (int): Output channels (teacher feature).
            mid_channels (int, optional): Intermediate channels. Defaults to out_channels.
        """
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        """Forward pass through alignment layers."""
        return self.align(x)


class ConvBNAligner(nn.Module):
    """Conv(BN+SiLU) -> Conv2d(linear). Maps student features to teacher feature dimensions using YOLO Conv module."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """Initialize ConvBNAligner.

        Args:
            in_channels (int): Input channels (student feature).
            out_channels (int): Output channels (teacher feature).
            mid_channels (int, optional): Intermediate channels. Defaults to out_channels.
        """
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.align = nn.Sequential(
            Conv(in_channels, mid_channels, k=1),  # Conv2d + BN + SiLU
            nn.Conv2d(mid_channels, out_channels, 1),  # Conv2d only (linear projection)
        )

    def forward(self, x):
        """Forward pass through alignment layers."""
        return self.align(x)


class ConvBNSiLUAligner(nn.Module):
    """Conv(BN+SiLU) -> Conv(BN+SiLU). Both layers include BN+SiLU to match teacher feature distribution."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """Initialize ConvBNSiLUAligner.

        Args:
            in_channels (int): Input channels (student feature).
            out_channels (int): Output channels (teacher feature).
            mid_channels (int, optional): Intermediate channels. Defaults to out_channels.
        """
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.align = nn.Sequential(
            Conv(in_channels, mid_channels, k=1),  # Conv2d + BN + SiLU
            Conv(mid_channels, out_channels, k=1),  # Conv2d + BN + SiLU
        )

    def forward(self, x):
        """Forward pass through alignment layers."""
        return self.align(x)


class MGDAligner(nn.Module):
    """Masked Generative Distillation connector (MGD, https://arxiv.org/abs/2205.01529).

    Randomly masks spatial positions of the student feature, then makes a small conv block
    *generate* the teacher feature from what survives. Structure follows the reference
    MGDConnector: an optional 1x1 projection (only when channel counts differ), then
    Conv3x3 -> ReLU -> Conv3x3 at teacher width.

    Unlike the other aligners here this one carries real capacity — the generation block is
    roughly 11x the parameters of ConvBNSiLUAligner at the same points. That is part of what
    MGD *is*, not an implementation choice, so an MGD run differs from an MSE/PKD run in three
    ways at once: projection type, masking, and the generation block. Resolve it with a
    lambda_mgd=0 control run, which keeps the block but disables masking (README §4).
    """

    def __init__(self, in_channels, out_channels, lambda_mgd=0.65, mask_on_channel=False):
        """Initialize MGDAligner.

        Args:
            in_channels (int): Input channels (student feature).
            out_channels (int): Output channels (teacher feature).
            lambda_mgd (float): Fraction of positions to mask out. 0 disables masking entirely.
            mask_on_channel (bool): Mask whole channels (N,C,1,1) instead of spatial positions (N,1,H,W).
        """
        super().__init__()
        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        # 채널이 같으면 투영을 두지 않는다 — 레퍼런스 구현과 동일하게 간다.
        self.align = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.generation = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        """Project, mask (training only), then generate the teacher feature."""
        if self.align is not None:
            x = self.align(x)
        # eval 에서는 마스킹하지 않는다. 이 aligner 는 학습 루프에서만 호출되지만 EMA 사본이 .eval() 이라 방어한다.
        if self.training and self.lambda_mgd > 0:
            n, c, h, w = x.shape
            shape = (n, c, 1, 1) if self.mask_on_channel else (n, 1, h, w)
            # lambda_mgd 는 "가려지는" 비율이다 — keep 확률이 1 - lambda_mgd 가 되도록 만든다.
            # 레퍼런스는 CPU 에서 만든 뒤 .to(device) 하지만, 여기서는 바로 올려 배치마다의 H2D 복사를 없앤다.
            keep = (torch.rand(shape, device=x.device, dtype=x.dtype) <= 1 - self.lambda_mgd).to(x.dtype)
            x = x * keep
        return self.generation(x)


class MultiScaleAligner(nn.Module):
    """Manages per-distillation-point ConvAligners via ModuleList."""

    def __init__(self, student_channels, teacher_channels, aligner_cls=ConvAligner):
        """Initialize MultiScaleAligner.

        Args:
            student_channels (list[int]): Channel counts for each student distillation point.
            teacher_channels (list[int]): Channel counts for each teacher distillation point.
            aligner_cls (type): Aligner class to use per point. Defaults to ConvAligner.
        """
        super().__init__()
        assert len(student_channels) == len(teacher_channels), (
            f"student ({len(student_channels)}) and teacher ({len(teacher_channels)}) channel lists must have equal length"
        )
        self.aligners = nn.ModuleList(
            [aligner_cls(sc, tc) for sc, tc in zip(student_channels, teacher_channels)]
        )

    def forward(self, features):
        """Align each feature map through its corresponding aligner.

        Args:
            features (list[torch.Tensor]): Student feature maps, one per distillation point.

        Returns:
            list[torch.Tensor]: Aligned feature maps matching teacher dimensions.
        """
        return [aligner(feat) for aligner, feat in zip(self.aligners, features)]
