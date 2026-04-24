# python scripts/neu_det/run_yolo26n_kd_neu_det.py
"""NEU-DET KD training — yolo26n student with yolo26s teacher.

ConvBNSiLUAligner로 backbone/neck → Detect head 내부 cv2/cv3 feature 정렬.
YOLO26는 end2end=True 기본값이지만, one2many 경로(cv2/cv3)만 KD 대상으로
삼으며 one2one 분기는 x.detach() 입력이라 backbone/neck 학습과 분리됨.
"""

from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(
        overrides={
            "model": "yolo26n.pt",
            "data": "NEU-DET.yaml",
            "epochs": 200,
            "batch": 32,
            "imgsz": 640,
            "workers": 4,
            "project": "neu_det",
            "name": "kd_convbnsilu_yolo26n_neu_det_200epoch",
            "patience": 50,
            "exist_ok": True,
            "device": 0,
            "distill_cfg": "ultralytics/cfg/distill_cfg_yolo26.yaml",
        }
    )
    trainer.train()
