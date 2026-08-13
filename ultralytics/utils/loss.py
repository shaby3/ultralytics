# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]


class PKDLoss(nn.Module):
    """Pearson-correlation feature distillation loss (PKD, https://arxiv.org/abs/2207.02039).

    Normalizes each channel to zero mean / unit variance before MSE, which makes the loss
    scale-free: the result equals `1 - r` where r is the Pearson correlation between the two
    feature maps. Plain MSE instead scales with the variance of the distillation point, which
    is why per-position KD weights have to be hand-normalized (see README §4). PKD removes that.

    The trade-off: standardization discards activation magnitude entirely, so only the pattern
    is distilled, not how strongly the teacher fires.

    Takes a single tensor pair — KDFeatureLoss already loops over distillation points, averages
    them, and detaches the teacher, and the trainer applies the KD weight. The reference
    implementation folds all three in; copying it wholesale would double-count.
    """

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Compute 1 - Pearson correlation between one student/teacher feature pair.

        Args:
            student_feat (torch.Tensor): Aligned student feature, shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature, same shape.

        Returns:
            torch.Tensor: Scalar loss in fp32.
        """
        assert student_feat.shape == teacher_feat.shape, (
            f"PKD requires matching shapes, got {tuple(student_feat.shape)} vs {tuple(teacher_feat.shape)}. "
            "The aligner should already match channels, and both hooks sit at the same stride."
        )
        # mse(standardized) / 2 == 1 - r:  E[(x-y)^2] = 1 - 2r + 1 = 2(1 - r)
        return F.mse_loss(self._norm(student_feat), self._norm(teacher_feat)) / 2

    @staticmethod
    def _norm(feat: torch.Tensor) -> torch.Tensor:
        """Standardize each channel over the (N, H, W) axes to zero mean and unit variance."""
        n, c, h, w = feat.shape
        # fp32 로 올린다 — KD forward 가 autocast 안이라 aligner conv 출력이 fp16 으로 들어온다.
        # 그대로 두면 N*H*W(P3 에서 10만개 이상) 누적에서 결과가 ~2e-4 흔들린다. 학습을 망칠 크기는
        # 아니지만 되돌리는 비용이 사실상 없고, 이 값이 weight 정규화의 기준이 되므로 재현성 쪽에 둔다.
        feat = feat.permute(1, 0, 2, 3).reshape(c, -1).float()
        feat = (feat - feat.mean(dim=-1, keepdim=True)) / (feat.std(dim=-1, keepdim=True) + 1e-6)
        return feat.reshape(c, n, h, w).permute(1, 0, 2, 3)


class AMSELoss(nn.Module):
    """Teacher-attention-weighted MSE feature distillation loss.

    Not a published standalone method: this isolates one component of FGD (arXiv 2111.11837),
    whose head-position run collapsed in this repo (README §4). FGD's focal term bundled three
    things — GT fg/bg masks, teacher-attention weighting, and an attention-matching L1 — and this
    loss keeps only the weighting, to test whether that part alone survives at the head position.

    Weights come from the *teacher* feature only: spatial `H*W*softmax(mean_c|t|/T)` and channel
    `C*softmax(mean_hw|t|/T)`, identical to FGD's parameter-free attention. Both average to
    exactly 1 over their axes, so the weighted mean keeps the plain-MSE scale, stays inside this
    repo's mean-reduction convention, and reduces to `F.mse_loss` when teacher attention is
    uniform. Teacher-only weighting matters: weighting by student attention would let the student
    shrink its own attention where it disagrees instead of closing the gap.

    Takes a single tensor pair like PKDLoss — KDFeatureLoss loops over points and detaches the
    teacher, and the trainer applies the KD weight.
    """

    def __init__(self, temp: float = 0.5):
        """Initialize AMSELoss.

        Args:
            temp (float): Softmax temperature for both attentions. FGD paper default 0.5 —
                lower is spikier weighting, higher approaches plain MSE.
        """
        super().__init__()
        self.temp = temp

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Compute teacher-attention-weighted MSE for one student/teacher feature pair.

        Args:
            student_feat (torch.Tensor): Aligned student feature, shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature, same shape (detached upstream).

        Returns:
            torch.Tensor: Scalar loss in fp32.
        """
        assert student_feat.shape == teacher_feat.shape, (
            f"AMSE requires matching shapes, got {tuple(student_feat.shape)} vs {tuple(teacher_feat.shape)}. "
            "The aligner should already match channels, and both hooks sit at the same stride."
        )
        # fp32 로 올린다 — PKDLoss 와 같은 이유 (autocast 안 fp16 입력에서 softmax·대면적 mean 재현성)
        s, t = student_feat.float(), teacher_feat.float()
        n, c, h, w = t.shape
        value = t.abs()
        # FGD 의 get_attention 과 동일한 정규화 — 두 가중치 모두 자기 축에서 평균이 정확히 1
        w_s = (h * w * F.softmax((value.mean(dim=1, keepdim=True) / self.temp).view(n, -1), dim=1)).view(n, 1, h, w)
        w_c = (c * F.softmax(value.mean(dim=(2, 3)) / self.temp, dim=1)).view(n, c, 1, 1)
        # mean(w·(s−t)²) == mse(s·√w, t·√w) — 가중 평균 1 이라 attention 이 균등하면 일반 MSE 로 환원된다
        return (w_s * w_c * (s - t) ** 2).mean()


class QMSEFeatureLoss(nn.Module):
    """Prediction-quality-weighted MSE feature distillation loss, branch-specific masks.

    Follow-up to AMSE (which weighted by activation magnitude and lost to plain MSE by
    -0.66pt): here the weighting criterion is the *quality of the teacher's predictions*.
    Each distillation point gets the mask matching what its branch predicts — box-branch
    (cv2) points are weighted by the IoU between the teacher's decoded box at each position
    and the best-overlapping GT, cls-branch (cv3) points by the teacher's max class score.
    The cls/reg mask split is borrowed from PGD (https://arxiv.org/abs/2203.05469), but this
    is NOT a PGD port: PGD selects top-k positions by s^(1-a)*IoU^a, fits a 2D Gaussian over
    them, and applies FGD-style fg/bg+attention losses. Here the component masks are used
    directly as all-pixel soft weights — no top-k, no Gaussian, no fg/bg, no attention.

    Masks are built from teacher predictions (+ GT for IoU) only, under no_grad — constants
    the student cannot game. Each mask is normalized to mean 1 per image/point, so the loss
    stays a weighted mean inside this repo's mean-reduction convention and reduces to plain
    MSE for a uniform mask. Parameter-free, but unlike the losses wrapped by KDFeatureLoss it
    needs the batch (GT), the teacher's decoded predictions, and the per-point branch mapping,
    so it replaces KDFeatureLoss as a container. No spatial-softmax and no channel weighting —
    the axes that degenerated for AMSE are absent by construction.
    """

    def __init__(self, mask_types: list[str]):
        """Initialize QMSEFeatureLoss.

        Args:
            mask_types (list[str]): Per-distillation-point mask kind, "iou" (box branch) or
                "score" (cls branch), in the same order as the configured layers.
        """
        super().__init__()
        assert mask_types and set(mask_types) <= {"iou", "score"}, f"invalid mask_types: {mask_types}"
        self.mask_types = list(mask_types)

    def forward(self, student_feats, teacher_feats, batch=None, teacher_preds=None):
        """Compute mean quality-weighted MSE across all distillation points.

        Args:
            student_feats (list[torch.Tensor]): Aligned student feature maps.
            teacher_feats (list[torch.Tensor]): Teacher feature maps (will be detached).
            batch (dict): Training batch; needs "img", "bboxes" (normalized xywh), "batch_idx".
            teacher_preds: Teacher eval-mode output; element 0 is (B, 4+nc, A) with decoded
                xywh pixel boxes and sigmoided class scores, anchors ordered P3->P4->P5 row-major.

        Returns:
            torch.Tensor: Scalar loss in fp32.
        """
        assert batch is not None and teacher_preds is not None, "QMSE needs the batch and teacher predictions"
        assert len(self.mask_types) == len(student_feats), (
            f"mask_types ({len(self.mask_types)}) and distillation points ({len(student_feats)}) mismatch"
        )
        y = (teacher_preds[0] if isinstance(teacher_preds, (tuple, list)) else teacher_preds).float()
        device = y.device
        score_all = y[:, 4:].amax(dim=1)  # (B, A) — 클래스 max. teacher 가 COCO(nc=80)면 80클래스 기준 확신도다
        pred_xyxy = xywh2xyxy(y[:, :4].permute(0, 2, 1))  # (B, A, 4) 픽셀 xyxy

        # 이미지별 GT 를 픽셀 xyxy 로, 위치별 IoU = 예측 박스 vs 이미지 내 전체 GT 의 max
        img_h, img_w = batch["img"].shape[-2:]
        gt = xywh2xyxy(batch["bboxes"].to(device).float()) * torch.tensor(
            [img_w, img_h, img_w, img_h], device=device
        )
        batch_idx = batch["batch_idx"].to(device).view(-1)
        iou_all = torch.zeros_like(score_all)
        for i in range(score_all.shape[0]):
            g = gt[batch_idx == i]
            if len(g):
                # (A,1,4) x (1,G,4) -> (A,G) — 위치당 최고 IoU 하나. 어느 GT 인지는 쓰지 않는다 (할당이 아니라 품질)
                iou_all[i] = bbox_iou(pred_xyxy[i].unsqueeze(1), g.unsqueeze(0), xywh=False).squeeze(-1).amax(dim=1)
            # GT 없는 이미지는 0 으로 남는다 — 아래 정규화 가드가 균등 마스크로 바꾼다

        # anchor 배열의 스케일별 구간: feature (H,W) 큰 순서(P3->P4->P5)가 make_anchors 순서와 같다
        sizes = sorted({tuple(f.shape[-2:]) for f in teacher_feats}, key=lambda s: -(s[0] * s[1]))
        offsets, start = {}, 0
        for h, w in sizes:
            offsets[(h, w)] = (start, start + h * w)
            start += h * w
        assert start == score_all.shape[1], f"anchor 수 불일치: 지점 {start} vs teacher 출력 {score_all.shape[1]}"

        loss = 0
        for mask_type, sf, tf in zip(self.mask_types, student_feats, teacher_feats):
            n, _, h, w = tf.shape
            s0, s1 = offsets[(h, w)]
            mask = (iou_all if mask_type == "iou" else score_all)[:, s0:s1].view(n, 1, h, w)
            # 이미지별 평균 1 정규화 — 가중 평균이 되어 균등 마스크면 일반 MSE 로 환원된다.
            # 평균이 0 근처(빈 GT 이미지의 IoU 마스크)면 균등 마스크로 대체한다.
            mean = mask.mean(dim=(2, 3), keepdim=True)
            mask = torch.where(mean > 1e-6, mask / mean.clamp(min=1e-6), torch.ones_like(mask))
            loss = loss + (mask * (sf.float() - tf.detach().float()) ** 2).mean()
        return loss / len(student_feats)


class KDFeatureLoss:
    """Feature-level Knowledge Distillation loss.

    Computes the mean loss across all distillation points between aligned student features
    and teacher features. Default loss function is MSE.
    """

    def __init__(self, loss_fn=None):
        """Initialize KDFeatureLoss.

        Args:
            loss_fn (nn.Module, optional): Loss function to use. Defaults to nn.MSELoss(reduction="mean").
        """
        self.loss_fn = loss_fn or nn.MSELoss(reduction="mean")

    def __call__(self, student_feats, teacher_feats, batch=None, teacher_preds=None):
        """Compute mean feature distillation loss across all distillation points.

        Args:
            student_feats (list[torch.Tensor]): Aligned student feature maps.
            teacher_feats (list[torch.Tensor]): Teacher feature maps (will be detached).
            batch (dict, optional): Ignored. Accepted so the trainer can call every KD loss with
                the same signature — QMSEFeatureLoss actually consumes it for GT boxes.
            teacher_preds (optional): Ignored. Same reason — QMSEFeatureLoss uses the teacher's
                decoded predictions for its quality masks.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        loss = sum(self.loss_fn(sf, tf.detach()) for sf, tf in zip(student_feats, teacher_feats))
        return loss / len(student_feats)
