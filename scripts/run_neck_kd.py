# python scripts/run_neck_kd.py
"""Neck-level KD training — yolov8n student with yolov8s teacher on VOC.

Distills neck output features (P3/P4/P5) instead of head-internal features.
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
            "name": "neck_kd_yolov8n",
            "exist_ok": True,
            "device": 0,
            "distill_cfg": "ultralytics/cfg/distill_neck_cfg.yaml",
        }
    )
    trainer.train()
