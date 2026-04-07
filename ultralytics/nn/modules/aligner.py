# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Feature alignment modules for knowledge distillation."""

import torch.nn as nn


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
