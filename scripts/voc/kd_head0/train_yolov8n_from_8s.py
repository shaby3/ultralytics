# python scripts/voc/kd_head0/train_yolov8n_from_8s.py
"""KD training — yolov8n student, yolov8s(VOC 학습본) teacher, Detect head 1번째 conv 에서 증류.

Phase 1(증류 위치 비교) 3런 중 하나. teacher·aligner·loss 는 세 런 공통이고
layers 와 weight 만 다르다 — ultralytics/cfg/distill_head0_from_8s_voc.yaml 참조.
weight 10 은 위치별 kd_loss 실측을 head1 기준으로 정규화한 값이다 (README §4).
head0 과 head1 은 출력 채널이 같아 aligner 까지 동일하다 — 세 수준 중 가장 깨끗한 비교.

1에폭 실측 11.0분 → 100에폭 약 18.3시간 (RTX 4050 6GB, batch 32).
에폭 말 검증(batch 64) 구간에서 peak 가 물리 VRAM 을 넘지만 시스템 RAM 스필로 완주한다 (README §4).

결과: runs/detect/voc/kd_head0/yolov8n_from_8s/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head0/yolov8n_from_8s"

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
            "batch": 32,  # baseline n 과 동일하게 고정 — 이게 깨지면 KD 효과와 batch 효과가 교락된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head0_from_8s_voc.yaml",
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
