# neck_kd_yolov8n

## 개요
Neck-level Feature KD 실험. Teacher(yolov8s)의 Neck output(P3/P4/P5)을 Student(yolov8n)로 증류.

## 세팅
| 항목 | 값 |
|------|-----|
| model (student) | yolov8n.pt |
| teacher | yolov8s.pt |
| data | VOC.yaml |
| epochs | 50 |
| batch | 32 |
| imgsz | 640 |
| optimizer | auto |
| device | 0 |
| amp | true |
| pretrained | true |
| distill_cfg | `ultralytics/cfg/distill_neck_cfg.yaml` |

## KD 설정
| 항목 | 값 |
|------|-----|
| KD 레이어 | layer 15 (P3/8), layer 18 (P4/16), layer 21 (P5/32) — Neck output 3개 |
| aligner | ConvAligner |
| loss | MSE |
| weight | 1.0 |

## 결과 (19/50 에폭)
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| 0.757 | 0.537 | 0.763 | 0.669 |
