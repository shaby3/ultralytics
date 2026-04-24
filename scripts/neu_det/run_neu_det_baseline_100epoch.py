# python scripts/run_neu_det_baseline.py
"""NEU-DET baseline training — yolo26n on surface defect dataset without KD."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    model.train(
        data="NEU-DET.yaml",
        epochs=100,
        batch=32,
        imgsz=640,
        workers=4,
        project="neu_det",
        name="baseline_yolo26n_neu_det_100epoch",
        exist_ok=True,
        device=0,
    )
