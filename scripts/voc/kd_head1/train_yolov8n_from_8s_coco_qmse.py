# python scripts/voc/kd_head1/train_yolov8n_from_8s_coco_qmse.py
"""KD training — yolov8n student, yolov8s COCO teacher, head `.1`, **loss = QMSE**.

AMSE(-0.66pt) 후속 ablation 런. train_yolov8n_from_8s_coco.py 에서 **loss 와 weight 만** 바꾼
통제 런이다 — 위치·teacher·aligner·epochs·batch 가 모두 같아 "가중 기준" 한 축만 갈린다:
균등(MSE) vs activation 크기(AMSE, 패배) vs **예측 품질(QMSE, 이번)**.

QMSE 는 branch 별 품질 마스크 가중 MSE 다 — box 분기(cv2)는 teacher 예측 박스 vs GT 의
max IoU, cls 분기(cv3)는 teacher 클래스 max score. PGD 의 cls/reg 분리를 차용한 자체 변형
(top-k·가우시안·fg/bg·attention 없음). ultralytics/cfg/distill_head1_from_8s_coco_qmse.yaml 참조.

비교 대상: runs/detect/voc/kd_head1/yolov8n_from_8s_coco/ (loss mse, mAP50-95 0.6496)
실질 선두: train_yolov8n_from_8s_coco_pkd.py (0.6533)

**조기 판정 기준:** ep1 이 역대 초반 밴드(0.39~0.45) 밖이면 ep10 전에 중단을 검토한다.

100에폭 약 13시간 (RTX 4050 6GB, batch 16). 마스크 계산은 teacher 의 기존 eval 출력을
재사용하고 이미지당 IoU 한 번이라 비용이 MSE 와 거의 같다 — 실측치는 프로브 참조.

결과: runs/detect/voc/kd_head1/yolov8n_from_8s_coco_qmse/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8s_coco_qmse"

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
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8s_coco_qmse.yaml",
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
