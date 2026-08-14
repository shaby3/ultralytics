# python scripts/voc/kd_head1/train_yolov8n_from_8s_coco_pqmse.py
"""KD training — yolov8n student, yolov8s COCO teacher, head `.1`, **loss = PQMSE**.

QMSE(기법 축 선두, 0.6564) 후속 런. train_yolov8n_from_8s_coco.py 에서 **loss 와 weight 만** 바꾼
통제 런이다 — 위치·teacher·aligner·epochs·batch 가 모두 같다. QMSE 와는 마스크 구성 두 축이
갈린다: 분기별 성분 마스크 -> PGD 식 쌍 결합 마스크(score_GT클래스^0.5 x IoU^2, 6지점 공통),
클래스 max -> GT 클래스 score. ultralytics/cfg/distill_head1_from_8s_coco_pqmse.yaml 참조.

비교 대상: runs/detect/voc/kd_head1/yolov8n_from_8s_coco_qmse/ (0.6564) — 이번 런의 실질 기준.
그 외: mse 0.6496, pkd 0.6533.

**조기 판정 기준:** ep1 이 역대 초반 밴드(0.39~0.45) 밖이면 ep10 전에 중단을 검토한다.

100에폭 약 13시간 (RTX 4050 6GB, batch 16). 마스크 계산은 QMSE 와 같은 경로(기존 teacher eval
출력 재사용)라 비용이 거의 같다.

결과: runs/detect/voc/kd_head1/yolov8n_from_8s_coco_pqmse/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8s_coco_pqmse"

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
            "batch": 16,  # 기법 런이 전부 16 이어야 한다 — 다르면 기법 x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8s_coco_pqmse.yaml",
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
