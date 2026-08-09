# python scripts/voc/kd_head1/train_yolov8n_from_8s_coco_mgd.py
"""KD training — yolov8n student, yolov8s COCO teacher, head `.1`, **기법 = MGD**.

Phase 3(KD 기법 비교) 런. train_yolov8n_from_8s_coco.py 에서 aligner 와 weight 만 바꿨다 —
위치·teacher·loss·epochs·batch 는 모두 같다. MGD 의 본체는 loss 가 아니라 aligner 자리에 있어서
loss 는 mse 그대로다 (ultralytics/cfg/distill_head1_from_8s_coco_mgd.yaml 참조).

비교 대상: runs/detect/voc/kd_head1/yolov8n_from_8s_coco/ (loss mse, mAP50-95 0.6496)
같은 축의 나머지: train_yolov8n_from_8s_coco_pkd.py

**해석 주의 — 이 런은 한 축만 다르지 않다.** MSE·PKD 런 대비 정렬부(ConvBNSiLU -> 1x1 Conv)·
랜덤 마스킹·생성 블록 셋이 함께 바뀌고, 증류 경로 파라미터가 100,608 -> 1,132,032 (11.3배)가 된다.
MGD 라는 방법의 정의라 피할 수 없다. 이기면 aligner_args 의 lambda_mgd 를 0 으로 둔 런
(생성 블록은 두고 마스킹만 끔)을 추가로 돌려 마스킹 순효과를 가른다. README §4.

100에폭 13시간 이상 (RTX 4050 6GB, batch 16). 6지점의 3x3 conv 두 층이 얹혀 MSE 런보다 느리다 —
실측치는 scripts/voc/probe_kd_scale.py 결과 참조.

결과: runs/detect/voc/kd_head1/yolov8n_from_8s_coco_mgd/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8s_coco_mgd"

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
            "batch": 16,  # 기법 3런이 전부 16 이어야 한다 — 다르면 기법 x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8s_coco_mgd.yaml",
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
