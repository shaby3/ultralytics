"""NEU-DET baseline test — yolo26n best.pt evaluated on val+test combined (360 images, 60/class)."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/neu_det/train/baseline_yolo26n_neu_det_200epoch/weights/best.pt")
    model.val(
        data="NEU-DET-valtest.yaml",
        split="test",
        imgsz=640,
        batch=32,
        workers=4,
        project="neu_det/test",
        name="baseline_yolo26n_neu_det_200epoch_valtest",
        exist_ok=True,
        device=0,
    )
