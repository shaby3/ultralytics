"""NEU-DET baseline test — yolo26l best.pt evaluated on held-out test split (180 images, 30/class, seed=42)."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/neu_det/train/baseline_yolo26l_neu_det_200epoch/weights/best.pt")
    model.val(
        data="NEU-DET.yaml",
        split="test",
        imgsz=640,
        batch=32,
        workers=4,
        project="neu_det/test",
        name="baseline_yolo26l_neu_det_200epoch_test",
        exist_ok=True,
        device=0,
    )
