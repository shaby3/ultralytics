# python scripts/gc10_det/kd/run_yolo26n_kd_gc10_det.py
"""GC10-DET KD training — yolo26n student with yolo26m baseline teacher.

Cold-start: student는 COCO pretrained yolo26n.pt에서 시작.
Teacher는 baseline_yolo26m_gc10_det_200epoch/weights/best.pt (GC10-DET fine-tuned).
ConvBNSiLUAligner로 Detect head 내부 cv2/cv3 feature 정렬.
YOLO26 end2end=True 환경에서 cv2/cv3(one2many) 경로만 KD 대상이며 one2one 분기는
x.detach() 입력이라 backbone/neck 학습과 분리됨.
"""

from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(
        overrides={
            "model": "yolo26n.pt",
            "data": "GC10-DET.yaml",
            "epochs": 200,
            "batch": 32,
            "imgsz": 640,
            "workers": 4,
            "project": "gc10_det/kd",
            "name": "kd_convbnsilu_yolo26n_gc10_det_200epoch",
            "patience": 50,
            "exist_ok": True,
            "device": 0,
            "distill_cfg": "ultralytics/cfg/distill_cfg_yolo26_gc10_det.yaml",
        }
    )
    trainer.train()
