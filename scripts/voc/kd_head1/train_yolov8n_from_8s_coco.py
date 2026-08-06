# python scripts/voc/kd_head1/train_yolov8n_from_8s_coco.py
"""KD training — yolov8n student, yolov8s **COCO-pretrained** teacher, Detect head 2번째 conv.

Phase 2(Q1: teacher 출처) 런. train_yolov8n_from_8s.py 에서 **teacher 만** 바꾼 통제 런이다 —
위치·aligner·weight·epochs·batch 가 모두 같아야 teacher 출처 하나의 효과가 분리된다.
설정을 하나라도 건드리면 Q1 이 교락되므로, 비교 대상과 함께 바꿀 것.

비교 대상: runs/detect/voc/kd_head1/yolov8n_from_8s/ (teacher s-VOC, mAP50-95 0.6342)

이 런을 먼저 도는 이유: 이전 회차(50에폭)의 COCO-teacher 런이 절반의 에폭으로 0.6428 을 냈다
(runs/detect/voc/kd_head1/yolov8n_from_8s_coco_50ep/). teacher 축이 예상보다 클 수 있어서,
남은 위치 스윕(neck·head0)을 돌리기 전에 teacher 부터 확정한다. README §6 참조.

1에폭 실측 7.8분 → 100에폭 약 13시간 (RTX 4050 6GB, batch 16).

결과: runs/detect/voc/kd_head1/yolov8n_from_8s_coco/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8s_coco"

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
    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(
        overrides={
            "model": "yolov8n.pt",  # baseline n 과 같은 COCO pretrained 출발점
            "data": "VOC.yaml",
            "project": PROJECT,
            "name": "train",
            "epochs": 100,
            "patience": 30,
            "batch": 16,  # _from_8s 런과 동일해야 한다 — 다르면 Q1 이 teacher x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8s_coco.yaml",
        }
    )
    trainer.train()

    # early stopping 이 걸리면 last 와 best 가 달라지므로 best 로 한 번 더 평가한다
    r = YOLO(trainer.best).val(project=PROJECT, name="val", exist_ok=True, **VAL_ARGS)

    # val/ 은 train/ 과 달리 지표를 파일로 남기지 않는다 (curve PNG 만 저장) — 직접 기록한다. README §5
    out = Path(r.save_dir)
    out.joinpath("results.csv").write_text(r.to_csv(), encoding="utf-8")  # 클래스별 P/R/F1/mAP
    out.joinpath("metrics.json").write_text(
        json.dumps({**r.results_dict, "speed": r.speed, "val_args": VAL_ARGS}, indent=2),
        encoding="utf-8",
    )
