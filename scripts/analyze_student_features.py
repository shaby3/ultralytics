# python scripts/analyze_student_features.py
"""Feature map analysis for a distilled student model.

Produces three sets of visualizations (histogram + channel-mean attention map):
    1) Neck output vs Head intermediate feature (per scale)
    2) cls branch (cv3) vs reg branch (cv2) feature (per scale)
    3) Head feature pre-aligner vs post-aligner (per distillation point)

Target experiment: runs/detect/kd_convbnsilu_yolov8n (ConvBNSiLUAligner, VOC).
Input image: ultralytics/assets/bus.jpg
Output: runs/detect/kd_convbnsilu_yolov8n/feature_analysis/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

# --- Configuration ---
EXP_DIR = Path("runs/detect/kd_convbnsilu_yolov8n")
STUDENT_CKPT = EXP_DIR / "weights" / "best.pt"
ALIGNER_CKPT = EXP_DIR / "weights" / "aligner_last.pt"
IMAGE_PATH = Path("ultralytics/assets/bus.jpg")
OUT_DIR = EXP_DIR / "feature_analysis"
IMGSZ = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_LAYERS = [
    "model.22.cv2.0.1",  # reg P3
    "model.22.cv2.1.1",  # reg P4
    "model.22.cv2.2.1",  # reg P5
    "model.22.cv3.0.1",  # cls P3
    "model.22.cv3.1.1",  # cls P4
    "model.22.cv3.2.1",  # cls P5
]
SCALE_NAMES = ["P3", "P4", "P5"]


# --- Loading ---


def load_student_and_aligner():
    """Load the distilled student DetectionModel and the trained MultiScaleAligner."""
    yolo = YOLO(str(STUDENT_CKPT))
    det_model = yolo.model.to(DEVICE).eval()

    ckpt = torch.load(str(ALIGNER_CKPT), map_location=DEVICE, weights_only=False)
    aligner = ckpt["aligner"].float().to(DEVICE).eval()
    return det_model, aligner


def preprocess_image(path: Path, imgsz: int) -> torch.Tensor:
    """Read image, letterbox to imgsz, convert to CHW float tensor in [0,1] on DEVICE."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True)
    img_bgr = lb(image=img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE)


# --- Hooks ---


class _FeatureHook:
    """Append forward output to a storage list."""

    def __init__(self, storage: list):
        self.storage = storage

    def __call__(self, module, inputs, output):
        self.storage.append(output)


class _DetectPreHook:
    """Capture the list input to Detect (i.e. neck outputs P3/P4/P5)."""

    def __init__(self, storage: list):
        self.storage = storage

    def __call__(self, module, inputs):
        x = inputs[0]
        self.storage.append([t.detach() for t in x])


def run_forward(det_model, image: torch.Tensor):
    """Run a single forward pass and collect neck + head features via hooks.

    Returns:
        neck_feats (list[Tensor]): 3 tensors, one per scale.
        head_feats (list[Tensor]): 6 tensors, order matching HEAD_LAYERS.
    """
    neck_storage: list = []
    head_storage: list = []

    detect_module = det_model.model[-1]
    pre_handle = detect_module.register_forward_pre_hook(_DetectPreHook(neck_storage))

    head_handles = []
    head_hook = _FeatureHook(head_storage)
    for spec in HEAD_LAYERS:
        mod = det_model.get_submodule(spec)
        head_handles.append(mod.register_forward_hook(head_hook))

    try:
        with torch.no_grad():
            det_model(image)
    finally:
        pre_handle.remove()
        for h in head_handles:
            h.remove()

    assert len(neck_storage) == 1, f"expected 1 neck capture, got {len(neck_storage)}"
    neck_feats = [t.detach() for t in neck_storage[0]]
    head_feats = [t.detach() for t in head_storage]
    assert len(neck_feats) == 3 and len(head_feats) == 6
    return neck_feats, head_feats


# --- Visualization helpers ---


def attention_map(feat: torch.Tensor) -> np.ndarray:
    """Channel-mean attention map (H, W) as numpy."""
    return feat.squeeze(0).float().mean(dim=0).cpu().numpy()


def _stats_caption(feat: torch.Tensor) -> str:
    f = feat.float()
    return f"mean={f.mean().item():.3f} std={f.std().item():.3f} min={f.min().item():.3f} max={f.max().item():.3f}"


def _plot_histogram(ax, feat: torch.Tensor, title: str):
    values = feat.float().flatten().cpu().numpy()
    ax.hist(values, bins=80, color="#4C78A8", alpha=0.85)
    ax.set_title(f"{title}\n{_stats_caption(feat)}", fontsize=9)
    ax.set_xlabel("activation")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)


def _plot_attention(ax, feat: torch.Tensor, title: str):
    amap = attention_map(feat)
    im = ax.imshow(amap, cmap="viridis")
    c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
    ax.set_title(f"{title}\nshape=({c}, {h}, {w})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Analysis 1: Neck vs Head (per scale) ---


def analyze_neck_vs_head(neck_feats, head_feats, out_dir: Path):
    """For each scale, compare neck output against head cv2/cv3 2nd-Conv output."""
    saved = []
    for i, scale in enumerate(SCALE_NAMES):
        neck = neck_feats[i]
        head_reg = head_feats[i]  # cv2.{i}.1
        head_cls = head_feats[i + 3]  # cv3.{i}.1

        fig, axes = plt.subplots(3, 2, figsize=(10, 11))
        fig.suptitle(f"Neck vs Head features @ {scale}", fontsize=13)

        _plot_histogram(axes[0, 0], neck, f"{scale} neck output — hist")
        _plot_attention(axes[0, 1], neck, f"{scale} neck output — attention")

        _plot_histogram(axes[1, 0], head_reg, f"{scale} head cv2.{i}.1 (reg) — hist")
        _plot_attention(axes[1, 1], head_reg, f"{scale} head cv2.{i}.1 (reg) — attention")

        _plot_histogram(axes[2, 0], head_cls, f"{scale} head cv3.{i}.1 (cls) — hist")
        _plot_attention(axes[2, 1], head_cls, f"{scale} head cv3.{i}.1 (cls) — attention")

        out_path = out_dir / f"{scale}.png"
        _save_figure(fig, out_path)
        saved.append(out_path)
    return saved


# --- Analysis 2: cls branch vs reg branch (per scale) ---


def analyze_cls_vs_reg(head_feats, out_dir: Path):
    saved = []
    for i, scale in enumerate(SCALE_NAMES):
        reg = head_feats[i]
        cls = head_feats[i + 3]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"reg (cv2) vs cls (cv3) @ {scale}", fontsize=13)

        _plot_histogram(axes[0, 0], reg, f"{scale} cv2.{i}.1 (reg) — hist")
        _plot_attention(axes[0, 1], reg, f"{scale} cv2.{i}.1 (reg) — attention")

        _plot_histogram(axes[1, 0], cls, f"{scale} cv3.{i}.1 (cls) — hist")
        _plot_attention(axes[1, 1], cls, f"{scale} cv3.{i}.1 (cls) — attention")

        out_path = out_dir / f"{scale}.png"
        _save_figure(fig, out_path)
        saved.append(out_path)
    return saved


# --- Analysis 3: Pre-aligner vs post-aligner (per distillation point) ---


def analyze_pre_vs_post_aligner(head_feats, aligner, out_dir: Path):
    with torch.no_grad():
        aligned_feats = aligner([f.float() for f in head_feats])

    saved = []
    for spec, pre, post in zip(HEAD_LAYERS, head_feats, aligned_feats):
        short = spec.replace("model.22.", "").replace(".", "_")  # e.g. cv2_0_1

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Pre vs Post aligner — {spec}", fontsize=13)

        _plot_histogram(axes[0, 0], pre, f"pre-aligner — hist")
        _plot_attention(axes[0, 1], pre, f"pre-aligner — attention")

        _plot_histogram(axes[1, 0], post, f"post-aligner — hist")
        _plot_attention(axes[1, 1], post, f"post-aligner — attention")

        out_path = out_dir / f"{short}.png"
        _save_figure(fig, out_path)
        saved.append(out_path)
    return saved


# --- Main ---


def main():
    print(f"[info] device={DEVICE}")
    print(f"[info] student ckpt: {STUDENT_CKPT}")
    print(f"[info] aligner ckpt: {ALIGNER_CKPT}")
    print(f"[info] image: {IMAGE_PATH}")

    det_model, aligner = load_student_and_aligner()
    image = preprocess_image(IMAGE_PATH, IMGSZ)
    print(f"[info] input shape: {tuple(image.shape)}")

    neck_feats, head_feats = run_forward(det_model, image)
    print(f"[info] neck shapes: {[tuple(f.shape) for f in neck_feats]}")
    print(f"[info] head shapes: {[tuple(f.shape) for f in head_feats]}")

    out_1 = OUT_DIR / "01_neck_vs_head"
    out_2 = OUT_DIR / "02_cls_vs_reg"
    out_3 = OUT_DIR / "03_head_vs_aligned"

    saved_1 = analyze_neck_vs_head(neck_feats, head_feats, out_1)
    saved_2 = analyze_cls_vs_reg(head_feats, out_2)
    saved_3 = analyze_pre_vs_post_aligner(head_feats, aligner, out_3)

    print("\n[done] saved figures:")
    for p in saved_1 + saved_2 + saved_3:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
