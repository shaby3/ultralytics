# python scripts/voc/kd_head1/train_yolov8n_from_8s_coco_fgd.py
"""KD training — yolov8n student, yolov8s COCO teacher, head `.1`, **기법 = FGD**.

Phase 3(KD 기법 비교) 런. train_yolov8n_from_8s_coco.py 에서 aligner·loss·weight 만 바꿨다 —
위치·teacher·epochs·batch 는 모두 같다. FGD 는 GT box 기반 전경/배경 분리(focal)와 GcBlock
전역 문맥(global)을 합친 기법이고, 학습 파라미터가 aligner 가 아니라 loss 모듈에 있다
(ultralytics/cfg/distill_head1_from_8s_coco_fgd.yaml 참조).

비교 대상: runs/detect/voc/kd_head1/yolov8n_from_8s_coco/ (loss mse, mAP50-95 0.6496)
같은 축의 나머지: train_yolov8n_from_8s_coco_pkd.py (0.6533) · train_yolov8n_from_8s_coco_mgd.py (0.6501)

**해석 주의 — off-label.** FGD 는 넥(FPN) feature 전용으로 설계·검증된 기법이다. 기법 축을
유지하려고 head`.1` 에 적용하지만 이 위치에 대한 논문 근거는 없다. 지는 경우 기법 탓인지
위치 탓인지 가를 수 없다. README §4.

100에폭 약 13시간 (RTX 4050 6GB, batch 16). FGD 는 지점마다 GT 마스크 루프가 얹혀 MSE 런보다
느릴 수 있다 — 실측치는 scripts/voc/probe_kd_scale.py 결과 참조.

결과: runs/detect/voc/kd_head1/yolov8n_from_8s_coco_fgd/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8s_coco_fgd"

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
            "batch": 16,  # 기법 4런이 전부 16 이어야 한다 — 다르면 기법 x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8s_coco_fgd.yaml",
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
