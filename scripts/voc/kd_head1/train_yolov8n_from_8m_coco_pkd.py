# python scripts/voc/kd_head1/train_yolov8n_from_8m_coco_pkd.py
"""KD training — yolov8n student, **yolov8m COCO teacher**, head `.1`, loss = PKD.

teacher 크기 축(Q3)의 두 번째 런. train_yolov8n_from_8s_coco_pkd.py 에서 **teacher 와 weight 만**
다르다 — 위치·loss·aligner·epochs·batch 가 같아 teacher 크기(s->m)만 분리된다.
(ultralytics/cfg/distill_head1_from_8m_coco_pkd.yaml 참조)

**왜 같은 축을 기법 2종으로 도는가:** m x QMSE 가 s x QMSE 에 ~0.6pt 뒤지는 궤적이라
"m 이 나쁜가" vs "QMSE x m 조합이 나쁜가"를 갈라야 한다. PKD 에서도 m 이 뒤지면 기법과 무관한
teacher 크기 효과(capacity gap)로 굳고, PKD 에서 m 이 이기면 기법 x teacher 크기 교호작용이다.

비교 4점: s x QMSE 0.6564 / s x PKD 0.6533 / m x QMSE(진행 중) / m x PKD(이 런).

**조기 판정 기준:** ep1 이 역대 초반 밴드(0.39~0.45) 밖이면 ep10 전에 중단을 검토한다.

100에폭 (RTX 4050 6GB, batch 16). m teacher forward 가 얹혀 s 런보다 느리다 — m x QMSE 실측
기준 에폭 ~11분, 완주 약 18시간. 실측치는 프로브(probe_kd_scale.json 의 pkd_8m) 참조.

결과: runs/detect/voc/kd_head1/yolov8n_from_8m_coco_pkd/{train,val}/
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/detect 를 붙이면 안 된다 — ultralytics 가 RUNS_DIR/<task>/ 아래로 합친다 (README §7.5)
PROJECT = "voc/kd_head1/yolov8n_from_8m_coco_pkd"

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
            "batch": 16,  # 비교 런들과 전부 16 이어야 한다 — 다르면 teacher x batch 교락이 된다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "distill_cfg": "ultralytics/cfg/distill_head1_from_8m_coco_pkd.yaml",
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
