# python scripts/voc/train/run_yolo26m_100epoch_patience10.py
"""VOC baseline training — yolo26m with COCO pretrained weights (converted for pruning-compatible C2f)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from convert_weights import convert

MODEL = "yolo26m"
SRC   = f"{MODEL}.pt"
CKPT  = f"{MODEL}_converted.pt"


def _ensure_converted() -> str:
    if Path(CKPT).exists():
        return CKPT
    src = attempt_download_asset(SRC)
    print(f"[convert] {src} → {CKPT}")
    convert(src, CKPT, f"{MODEL}.yaml")
    return CKPT


if __name__ == "__main__":
    ckpt = _ensure_converted()
    model = YOLO(ckpt)
    model.train(
        data="VOC.yaml",
        epochs=100,
        batch=8,
        imgsz=640,
        workers=2,
        project="voc/train",
        name="baseline_yolo26m_100epoch",
        patience=20,
        exist_ok=True,
        device=0,
    )
