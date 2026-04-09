# python scripts/run_phase6_kd_convbn.py
"""Phase 6 KD training — ConvBNAligner variant.

yolov8n student with yolov8s teacher on VOC, using ConvBNAligner
(Conv+BN+SiLU -> linear Conv2d) for student-teacher feature alignment.
"""

from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(
        overrides={
            "model": "yolov8n.pt",
            "data": "VOC.yaml",
            "epochs": 50,
            "batch": 32,
            "imgsz": 640,
            "workers": 4,
            # "project": "runs/detect_voc",
            "name": "kd_convbn_yolov8n",
            "exist_ok": True,
            "device": 0,
            "distill_cfg": "ultralytics/cfg/distill_cfg_convbn.yaml",
        }
    )
    trainer.train()
