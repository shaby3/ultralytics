# kd_convbnsilu_yolov8n

## 개요
Head-level Feature KD 실험 (aligner 변형). `kd_yolov8n`과 동일한 cv2/cv3 2nd conv 6개 레이어를 증류하되, aligner를 `ConvBNSiLUAligner`로 교체 — Conv+BN+SiLU 블록이 단순 Conv aligner 대비 KD에 미치는 영향 확인.

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
| distill_cfg | `ultralytics/cfg/distill_cfg_convbnsilu.yaml` |

## KD 설정
| 항목 | 값 |
|------|-----|
| KD 레이어 | Detect head cv2(box) 0/1/2번 2nd conv, cv3(cls) 0/1/2번 2nd conv (총 6개) |
| aligner | ConvBNSiLUAligner |
| loss | MSE |
| weight | 1.0 |

## 결과 (50/50 에폭 완료)
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| 0.843 | 0.643 | 0.807 | 0.777 |

> 다른 실험과의 비교는 [`../EXPERIMENTS.md`](../EXPERIMENTS.md) 참고.
