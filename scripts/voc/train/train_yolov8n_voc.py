# python scripts/voc/train/train_yolov8n_voc.py
"""Baseline training — yolov8n on Pascal VOC (KD 학생 모델).

KD 실험의 student baseline. COCO pretrained 가중치에서 fine-tune.
batch는 1에폭 실측(검증 포함)으로 확정한 값 — RTX 4050 6GB 기준.
optimizer=auto 는 100에폭에서 MuSGD 를 선택한다(iterations > 10000).
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        patience=30,
        batch=32,  # 1에폭 실측 고정: peak 3.68GB/6GB, 1ep≈8.9min (AutoBatch 추천 12는 과보수)
        imgsz=640,
        workers=2,
        name="voc_baseline_yolov8n",
        exist_ok=True,
        device=0,
        amp=True,
    )
