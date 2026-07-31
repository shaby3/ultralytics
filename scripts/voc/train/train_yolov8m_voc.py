# python scripts/voc/train/train_yolov8m_voc.py
"""Baseline training — yolov8m on Pascal VOC (KD 교사 모델, VOC 학습 버전).

baseline 지표이자 KD의 VOC-finetuned teacher로 사용.
batch는 1에폭 실측(검증 포함)으로 확정한 값 — RTX 4050 6GB 기준.
optimizer=auto 는 100에폭에서 MuSGD 를 선택한다(iterations > 10000).
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        patience=30,
        batch=8,  # 1에폭 실측 고정: peak 3.20GB/6GB, 1ep≈22.7min (AutoBatch 추천 2는 과보수)
        imgsz=640,
        workers=2,
        name="voc_baseline_yolov8m",
        exist_ok=True,
        device=0,
        amp=True,
    )
