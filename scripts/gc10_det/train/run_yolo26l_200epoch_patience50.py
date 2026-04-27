# python scripts/gc10_det/train/run_yolo26l_200epoch_patience50.py
"""GC10-DET baseline training — yolo26l on surface defect dataset without KD."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26l.pt")
    model.train(
        data="GC10-DET.yaml",
        epochs=200,
        batch=32,
        imgsz=640,
        workers=4,
        project="gc10_det/train",
        name="baseline_yolo26l_gc10_det_200epoch",
        patience=50,
        exist_ok=True,
        device=0,
    )
