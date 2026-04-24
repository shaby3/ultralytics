# python scripts/run_neu_det_baseline.py
"""NEU-DET baseline training — yolo26n on surface defect dataset without KD."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26m.pt")
    model.train(
        data="NEU-DET.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        workers=2,
        project="neu_det/train",
        name="baseline_yolo26m_neu_det_200epoch",
        patience=50,
        exist_ok=True,
        device=0,
    )
