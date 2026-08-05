# python scripts/voc/baseline/train_yolov8n.py
"""Baseline training — yolov8n on Pascal VOC (KD 학생 모델).

KD 실험의 student baseline. COCO pretrained 가중치에서 fine-tune.
batch는 1에폭 실측(검증 포함)으로 확정한 값 — RTX 4050 6GB 기준.
optimizer=auto 는 100에폭에서 MuSGD 를 선택한다(iterations > 10000).

결과: runs/detect/voc/baseline/yolov8n/{train,val}/
VOC.yaml 은 val 과 test 가 둘 다 images/test2007 이라 test split 은 돌리지 않는다.
"""

import json
from pathlib import Path

from ultralytics import YOLO

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/baseline/yolov8n"

# 평가 조건을 명시해 박는다. 기본값에 의존하면 업스트림이 바꿀 때 과거 결과를 재현할 수 없다. README §7.7
VAL_ARGS = dict(
    data="VOC.yaml",
    imgsz=640,
    batch=32,  # rect=True 라 mAP 에도 영향이 있다 — README §7.7
    workers=2,
    device=0,
    conf=0.001,  # detect val 기본값 (predict 는 0.25)
    iou=0.7,  # NMS
    max_det=300,
    half=False,
    rect=True,  # default.yaml 은 False 지만 val 은 True 가 맞다 — README §7.7
)

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
    r = YOLO(model.trainer.best).val(project=PROJECT, name="val", exist_ok=True, **VAL_ARGS)

    # val/ 은 train/ 과 달리 지표를 파일로 남기지 않는다 (curve PNG 만 저장) — 직접 기록한다. README §5
    out = Path(r.save_dir)
    out.joinpath("results.csv").write_text(r.to_csv(), encoding="utf-8")  # 클래스별 P/R/F1/mAP
    out.joinpath("metrics.json").write_text(
        json.dumps({**r.results_dict, "speed": r.speed, "val_args": VAL_ARGS}, indent=2),
        encoding="utf-8",
    )
