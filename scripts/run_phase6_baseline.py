# python scripts/run_phase6_baseline.py
"""Phase 6 TODO-16: Baseline training — yolov8n on VOC without KD."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="VOC.yaml",
        epochs=50,
        batch=32,
        imgsz=640,
        workers=4,
        # project="runs/detect_voc",
        name="baseline_yolov8n",
        exist_ok=True,
        device=0,
    )
