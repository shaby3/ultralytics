# 실험 성능 정리

YOLOv8n + VOC 기반 KD 실험 결과 모음. 각 실험의 상세 세팅은 해당 디렉토리의 `README.md` 참고.

## 공통 조건
| 항목 | 값 |
|------|-----|
| student | yolov8n.pt |
| teacher | yolov8s.pt (KD 실험 한정) |
| data | VOC.yaml |
| epochs | 50 |
| imgsz | 640 |
| optimizer | auto |
| device | 0 |
| amp | true |
| pretrained | true |

## 결과 요약
| 실험 | KD 위치 | batch | mAP50 | mAP50-95 | Precision | Recall |
|------|---------|-------|-------|----------|-----------|--------|
| [baseline_yolov8n](./baseline_yolov8n/) | — | 16 | 0.817 | 0.612 | 0.804 | 0.745 |
| [neck_kd_yolov8n](./neck_kd_yolov8n/) | Neck output P3/P4/P5 (layer 15/18/21) | 32 | 0.821 | 0.619 | 0.803 | 0.754 |
| [kd_yolov8n](./kd_yolov8n/) | Detect head cv2/cv3 2nd conv × 3 scale | 32 | **0.843** | **0.640** | **0.818** | **0.770** |

## 베이스라인 대비 개선폭 (%p)
| 실험 | ΔmAP50 | ΔmAP50-95 | ΔPrecision | ΔRecall |
|------|--------|-----------|------------|---------|
| neck_kd_yolov8n | +0.4 | +0.7 | -0.1 | +0.9 |
| kd_yolov8n (head) | **+2.6** | **+2.8** | **+1.4** | **+2.5** |

## 관찰
- **Head KD가 Neck KD를 명확히 상회.** Detect head의 cv2(box)/cv3(cls) feature를 증류할 때 Student가 Teacher의 task-specific representation을 더 직접적으로 학습.
- **Neck KD는 baseline 대비 개선이 있지만 미미하다.** P3/P4/P5 output feature는 task head 직전 단계라 정보량이 많지만, Student-Teacher 간 채널/공간 정렬만으로는 KD 효과가 제한적.
- Precision은 모든 실험에서 거의 동일(0.803~0.818), Recall이 주된 개선 영역.

## 업데이트 규칙
- 새 실험이 끝나면 이 파일의 "결과 요약" 표에 행을 추가한다.
- 각 실험 디렉토리의 `README.md`에는 **해당 실험 고유의 세팅/결과**만 기록하고, 비교는 이 파일에서 수행한다.
