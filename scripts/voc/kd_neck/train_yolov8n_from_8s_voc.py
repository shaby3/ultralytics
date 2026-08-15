# python scripts/voc/kd_neck/train_yolov8n_from_8s_voc.py
"""KD training — yolov8n student, yolov8s **VOC 학습본** teacher, **neck**(15/18/21), loss = MSE.

위치 × teacher 출처 2×2 를 채우는 런. 위치 순위(head`.1` > head`.0` > neck)는 지금까지
**s-COCO 에서만** 확인됐고, teacher 를 바꿔도 같은 순위인지는 검증되지 않았다 — greedy 탐색의
한계다 (README §1). 이 런이 마지막 칸을 메운다:

    |        | s-VOC      | s-COCO |
    | neck   | 이 런      | 0.6408 |
    | head.1 | 0.6342     | 0.6496 |

두 방향 모두 한 축 비교다 — train_yolov8n_from_8s_coco.py(neck)와는 teacher 만,
kd_head1/train_yolov8n_from_8s_voc.py 와는 위치만 다르다.

**읽는 법:** head`.1` − neck 격차가 s-COCO 에서 +0.88pt 였다. VOC teacher 에서도 head 가
우세하면 위치 결론이 teacher 와 무관하게 성립하고, 뒤집히면 위치 × teacher 교호작용이 있다는
뜻이라 "head 채택" 결론에 단서를 달아야 한다.

**조기 판정 기준:** ep1 이 역대 초반 밴드(0.39~0.45) 밖이면 ep10 전에 중단을 검토한다.

100에폭 약 13시간 (RTX 4050 6GB, batch 16).

결과: runs/detect/voc/kd_neck/yolov8n_from_8s/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
# 접미사 없는 variant = VOC 학습본 teacher (README §5 네이밍 규칙)
PROJECT = "voc/kd_neck/yolov8n_from_8s"

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
            "batch": 16,  # 비교 런들과 전부 16 이어야 한다 — 다르면 위치 x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_neck_from_8s_voc.yaml",
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
