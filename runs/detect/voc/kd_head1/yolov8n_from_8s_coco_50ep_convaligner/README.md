> **git 히스토리에서 복원한 이전 회차 실험이다.** 원래 경로 `runs/detect/kd_yolov8n`,
> 복원 출처 `8e45746f~1`. 아래 본문은 당시 기록 그대로이며, 아래 주의사항만 덧붙였다.
>
> **aligner 가 `ConvAligner`** 다 — 현재는 `ConvBNSiLUAligner` 로 고정했다(README §4).
> 같은 회차의 `../yolov8n_from_8s_coco_50ep/`(ConvBNSiLU)와 이 런의 차이가 곧 aligner 효과다.
> 그 외 teacher(COCO-pretrained) · 50에폭 · batch 32 는 위 런과 동일하다.

---

# kd_yolov8n

## 개요
Head-level Feature KD 실험. Teacher(yolov8s)의 Detect head cv2/cv3 브랜치 feature를 Student(yolov8n)로 증류.

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
| distill_cfg | `ultralytics/cfg/distill_cfg.yaml` |

## KD 설정
| 항목 | 값 |
|------|-----|
| KD 레이어 | Detect head cv2(box) 0/1/2번 2nd conv, cv3(cls) 0/1/2번 2nd conv (총 6개) |
| aligner | ConvAligner |
| loss | MSE |
| weight | 1.0 |

## 결과
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| 0.843 | 0.640 | 0.818 | 0.770 |
