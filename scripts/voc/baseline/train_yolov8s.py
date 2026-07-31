# python scripts/voc/baseline/train_yolov8s.py
"""Baseline training — yolov8s on Pascal VOC (KD 교사 모델, VOC 학습 버전).

baseline 지표이자 KD의 VOC-finetuned teacher로 사용.
batch는 1에폭 실측(검증 포함)으로 확정한 값 — RTX 4050 6GB 기준.
optimizer=auto 는 100에폭에서 MuSGD 를 선택한다(iterations > 10000).

결과: runs/detect/voc/baseline/yolov8s/{train,val}/
VOC.yaml 은 val 과 test 가 둘 다 images/test2007 이라 test split 은 돌리지 않는다.
"""

from ultralytics import YOLO

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/baseline/yolov8s"

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(
        data="VOC.yaml",
        project=PROJECT,
        name="train",
        epochs=100,
        patience=30,
        batch=16,  # 1에폭 실측 고정: peak 3.36GB/6GB, 1ep≈10.8min (AutoBatch 추천 4는 과보수)
        imgsz=640,
        workers=2,
        exist_ok=True,
        device=0,
        amp=True,
    )

    # early stopping 이 걸리면 last 와 best 가 달라지므로 best 로 한 번 더 평가한다
    YOLO(model.trainer.best).val(
        data="VOC.yaml",
        project=PROJECT,
        name="val",
        imgsz=640,
        batch=16,
        workers=2,
        exist_ok=True,
        device=0,
    )
