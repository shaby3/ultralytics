"""Phase 2 tests: KDFeatureLoss."""

import torch

from ultralytics.utils.loss import KDFeatureLoss


def test_identical_tensors_zero_loss():
    kd_loss = KDFeatureLoss()
    feats = [torch.randn(1, 64, 10, 10) for _ in range(3)]
    loss = kd_loss(feats, feats)
    assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"


def test_different_tensors_positive_loss():
    kd_loss = KDFeatureLoss()
    s_feats = [torch.zeros(1, 64, 10, 10) for _ in range(3)]
    t_feats = [torch.ones(1, 64, 10, 10) for _ in range(3)]
    loss = kd_loss(s_feats, t_feats)
    assert loss.item() > 0, f"Expected > 0, got {loss.item()}"


def test_teacher_detached():
    kd_loss = KDFeatureLoss()
    s = [torch.randn(1, 64, 10, 10, requires_grad=True)]
    t = [torch.randn(1, 64, 10, 10, requires_grad=True)]
    loss = kd_loss(s, t)
    loss.backward()
    assert s[0].grad is not None, "Student should have gradient"
    assert t[0].grad is None, "Teacher should be detached (no gradient)"


def test_multi_point_average():
    kd_loss = KDFeatureLoss()
    s_feats = [torch.zeros(1, 64, 10, 10), torch.zeros(1, 80, 10, 10)]
    t_feats = [torch.ones(1, 64, 10, 10), torch.ones(1, 80, 10, 10)]
    loss = kd_loss(s_feats, t_feats)
    # MSE of zeros vs ones = 1.0 per point, average = 1.0
    assert abs(loss.item() - 1.0) < 1e-5, f"Expected 1.0, got {loss.item()}"


if __name__ == "__main__":
    test_identical_tensors_zero_loss()
    test_different_tensors_positive_loss()
    test_teacher_detached()
    test_multi_point_average()
    print("All Phase 2 tests passed!")
