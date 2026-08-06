> **git 히스토리에서 복원한 이전 회차 실험이다.** 원래 경로 `runs/detect/baseline_yolov8n`,
> 복원 출처 `8e45746f~1`. 아래 본문은 당시 기록 그대로이며, 아래 주의사항만 덧붙였다.
>
> **현재 회차와 비교할 때 주의:**
> - **50에폭**이다. 현재 baseline(`../yolov8n/`)은 100에폭이고 LR 감쇠 스케줄이 다르다.
> - **batch 16**이다. 같은 회차의 KD 런들은 batch 32 였다 —
>   즉 이 회차의 baseline 대비 KD gain 에는 batch 교락이 섞여 있다.
> - 이 회차에는 `val/`(best.pt 재평가) 단계가 없었다. 아래 수치는 `train/` 마지막 에폭 값이다.

---

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
