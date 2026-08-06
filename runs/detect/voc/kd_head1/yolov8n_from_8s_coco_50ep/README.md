> **git 히스토리에서 복원한 이전 회차 실험이다.** 원래 경로 `runs/detect/kd_convbnsilu_yolov8n`,
> 복원 출처 `8e45746f~1`. 아래 본문은 당시 기록 그대로이며, 아래 주의사항만 덧붙였다.
>
> **현재 Phase 1(`../yolov8n_from_8s/`)과의 차이 — 이 셋이 겹쳐 있어 직접 비교는 불가능하다:**
> - **teacher 가 `yolov8s.pt` = COCO-pretrained** 다. 현재 Phase 1 은 s-VOC 학습본을 쓴다.
>   이 런이 현재 런보다 높다면 그건 Phase 2(Q1)가 답할 질문의 예고편이다.
> - **50에폭**(현재 100에폭) — LR 스케줄이 다르다.
> - **batch 32**(현재 16). accumulate 로 유효 batch 는 같고 BN 통계만 다르다.
>
> aligner 는 `ConvBNSiLUAligner` 로 현재와 같고, weight 1.0 도 현재 head1 기준값과 같다.
> 즉 **현재 Phase 1 세 런 중 head1 이 이 런과 가장 조건이 가깝다.**

---

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
