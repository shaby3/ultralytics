# python scripts/gc10_det/kd/run_yolo26n_kd_unified_gc10_det.py
"""GC10-DET KD training — yolo26n student, one2many + one2one 통합 증류.

Cold-start: student는 COCO pretrained yolo26n.pt에서 시작.
Teacher는 baseline_yolo26m_gc10_det_200epoch/weights/best.pt (GC10-DET fine-tuned).
ConvBNSiLUAligner로 Detect head의 cv2/cv3(one2many) + one2one_cv2/cv3(one2one) 총 12포인트 정렬.
one2one KD gradient는 one2one_cv2/cv3 가중치만 업데이트 (x.detach() 특성).
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
            "name": "kd_unified_yolo26n_gc10_det_200epoch",
            "patience": 50,
            "exist_ok": True,
            "device": 0,
            "distill_cfg": "ultralytics/cfg/distill_cfg_yolo26_gc10_det_unified.yaml",
        }
    )
    trainer.train()
