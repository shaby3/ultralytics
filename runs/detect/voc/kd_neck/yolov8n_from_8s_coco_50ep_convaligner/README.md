> **git 히스토리에서 복원한 이전 회차 실험이다.** 원래 경로 `runs/detect/neck_kd_yolov8n`,
> 복원 출처 `8e45746f~1`. 아래 본문은 당시 기록 그대로이며, 아래 주의사항만 덧붙였다.
>
> **이 런의 결과를 "neck 이 head 보다 나쁘다"로 읽으면 안 된다.** 두 가지가 겹쳐 있다:
> - **weight 1.0 이 neck 에서는 사실상 KD 를 끈 것과 같다.** 이 런의 최종 `train/kd_loss` 는
>   **0.038**, 같은 회차 head1 런은 **0.974** 로 25배 차이다. kd_loss 는 MSE 라 위치별
>   feature 분산에 비례하는데, 당시엔 weight 를 정규화하지 않았다.
>   현재 Phase 1 은 이 문제를 실측으로 보정했다 — neck weight 20 (README §4).
> - **aligner 가 `ConvAligner`** 다. 현재는 `ConvBNSiLUAligner` 로 고정.
>
> teacher(COCO-pretrained) · 50에폭 · batch 32 는 같은 회차의 head 런들과 동일하다.

---

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

## 결과 (50/50 에폭 완료)
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| 0.821 | 0.619 | 0.803 | 0.754 |

> 다른 실험과의 비교는 [`../EXPERIMENTS.md`](../EXPERIMENTS.md) 참고.
