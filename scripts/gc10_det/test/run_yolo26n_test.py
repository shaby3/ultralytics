"""GC10-DET baseline test — yolo26n best.pt evaluated on held-out test split (10% per class, seed=42)."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/gc10_det/train/baseline_yolo26n_gc10_det_200epoch/weights/best.pt")
    model.val(
        data="GC10-DET.yaml",
        split="test",
        imgsz=640,
        batch=32,
        workers=4,
        project="gc10_det/test",
        name="baseline_yolo26n_gc10_det_200epoch_test",
        exist_ok=True,
        device=0,
    )
