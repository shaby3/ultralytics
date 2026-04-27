# python scripts/gc10_det/train/run_yolo26m_200epoch_patience50.py
"""GC10-DET baseline training — yolo26m on surface defect dataset without KD."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26m.pt")
    model.train(
        data="GC10-DET.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        workers=2,
        project="gc10_det/train",
        name="baseline_yolo26m_gc10_det_200epoch",
        patience=50,
        exist_ok=True,
        device=0,
    )
