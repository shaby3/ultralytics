# python scripts/voc/baseline/train_yolov8n.py
"""Baseline training — yolov8n on Pascal VOC (KD 학생 모델).

KD 실험의 student baseline. COCO pretrained 가중치에서 fine-tune.
batch는 1에폭 실측(검증 포함)으로 확정한 값 — RTX 4050 6GB 기준.
optimizer=auto 는 100에폭에서 MuSGD 를 선택한다(iterations > 10000).

결과: runs/detect/voc/baseline/yolov8n/{train,val}/
VOC.yaml 은 val 과 test 가 둘 다 images/test2007 이라 test split 은 돌리지 않는다.
"""

from ultralytics import YOLO

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/baseline/yolov8n"

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="VOC.yaml",
        project=PROJECT,
        name="train",
        epochs=100,
        patience=30,
        batch=32,  # 1에폭 실측 고정: peak 3.68GB/6GB, 1ep≈8.9min (AutoBatch 추천 12는 과보수)
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
        batch=32,
        workers=2,
        exist_ok=True,
        device=0,
    )
