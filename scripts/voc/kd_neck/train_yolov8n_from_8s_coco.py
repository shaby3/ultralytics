# python scripts/voc/kd_neck/train_yolov8n_from_8s_coco.py
"""KD training — yolov8n student, yolov8s **COCO-pretrained** teacher, neck 출력(P3/P4/P5)에서 증류.

재정렬된 Phase 1(증류 위치 3수준) 중 하나. Phase 2 에서 COCO teacher 가 s-VOC 를
+1.53pt 로 이겨서, 위치 스윕 전체를 COCO teacher 위에서 돌기로 했다 (README §4·§6).

같은 위치 스윕의 나머지 둘 — teacher·aligner·loss 가 공통이고 layers 와 weight 만 다르다:
  scripts/voc/kd_head0/train_yolov8n_from_8s_coco.py
  scripts/voc/kd_head1/train_yolov8n_from_8s_coco.py  (완료, mAP50-95 0.6496)
weight 20 은 위치별 kd_loss 실측을 head1 기준으로 정규화한 값이다 —
ultralytics/cfg/distill_neck_from_8s_coco.yaml 및 README §4 참조.

teacher 만 다른 통제 런: train_yolov8n_from_8s.py (teacher s-VOC) — 아직 미실행.
설정을 하나라도 건드리면 위치 비교가 교락되므로, 세 런을 함께 바꿀 것.

head1 실측 7.8분/에폭(batch 16, peak 2.9G/6.1G) — README §7.8.
neck 은 KD 지점이 적어 이보다 빠르다. 100에폭 약 11~13시간.

결과: runs/detect/voc/kd_neck/yolov8n_from_8s_coco/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_neck/yolov8n_from_8s_coco"

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
            "batch": 16,  # 위치 스윕 3런이 전부 16 이어야 한다 — 다르면 위치 x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_neck_from_8s_coco.yaml",
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
