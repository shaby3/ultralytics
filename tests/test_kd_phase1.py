"""Phase 1 tests: Aligner modules and imports."""

import torch

from ultralytics.nn.modules import ConvAligner, MultiScaleAligner


def test_conv_aligner_shape():
    aligner = ConvAligner(64, 80)
    x = torch.randn(1, 64, 20, 20)
    out = aligner(x)
    assert out.shape == (1, 80, 20, 20)


def test_conv_aligner_mid_channels():
    aligner = ConvAligner(256, 512, mid_channels=128)
    x = torch.randn(2, 256, 10, 10)
    out = aligner(x)
    assert out.shape == (2, 512, 10, 10)


def test_multi_scale_aligner():
    student_ch = [64, 64, 64, 80, 80, 80]
    teacher_ch = [128, 128, 128, 160, 160, 160]
    ms = MultiScaleAligner(student_ch, teacher_ch)
    assert len(ms.aligners) == 6

    features = [torch.randn(1, sc, 10, 10) for sc in student_ch]
    aligned = ms(features)
    for af, tc in zip(aligned, teacher_ch):
        assert af.shape[1] == tc


def test_multi_scale_aligner_length_mismatch():
    try:
        MultiScaleAligner([64, 64], [128])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_gradient_flow():
    aligner = ConvAligner(64, 80)
    x = torch.randn(1, 64, 10, 10, requires_grad=True)
    out = aligner(x)
    out.sum().backward()
    for _, p in aligner.named_parameters():
        assert p.grad is not None


if __name__ == "__main__":
    test_conv_aligner_shape()
    test_conv_aligner_mid_channels()
    test_multi_scale_aligner()
    test_multi_scale_aligner_length_mismatch()
    test_gradient_flow()
    print("All Phase 1 tests passed!")
