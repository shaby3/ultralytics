# baseline_yolov8n

## 개요
KD 없이 YOLOv8n을 Pascal VOC 데이터셋으로 학습한 베이스라인 실험.

## 세팅
| 항목 | 값 |
|------|-----|
| model | yolov8n.pt |
| data | VOC.yaml |
| epochs | 50 |
| batch | 16 |
| imgsz | 640 |
| optimizer | auto |
| device | 0 |
| amp | true |
| pretrained | true |
| distill_cfg | 없음 |

## 결과
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| 0.817 | 0.612 | 0.804 | 0.745 |
