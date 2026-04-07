# models/yolo/ - YOLO 모델 패밀리

## 개요
핵심 YOLO 구현. YOLO, YOLOWorld, YOLOE 클래스와 태스크별 trainer/validator/predictor 트라이어드 포함.

## 태스크 시스템

| 태스크 | Trainer | Head | Loss |
|--------|---------|------|------|
| detect | `detect/train.py` | `Detect` | `v8DetectionLoss` |
| segment | `segment/train.py` | `Segment` | `v8SegmentationLoss` |
| classify | `classify/train.py` | `Classify` | `v8ClassificationLoss` |
| pose | `pose/train.py` | `Pose` | `v8PoseLoss` |
| obb | `obb/train.py` | `OBB` | `v8OBBLoss` |

태스크는 `model.py`의 `YOLO.task_map`으로 라우팅됨.

## 핵심 파일
- `model.py` — YOLO 클래스 (파일명 패턴에 따라 YOLOWorld/YOLOE로 자동 전환), YOLOWorld 클래스, YOLOE 클래스. `task_map` 프로퍼티 포함.
- `__init__.py` — YOLO, YOLOWorld, YOLOE re-export

## 새 태스크 추가 방법
1. `models/yolo/{newtask}/`에 `train.py`, `val.py`, `predict.py` 생성 (Base* 상속)
2. `nn/modules/head.py`에 헤드 추가
3. `utils/loss.py`에 손실 함수 추가
4. `cfg/__init__.py`의 TASKS set과 `models/yolo/model.py`의 `YOLO.task_map`에 등록

## 특수 변형
- **YOLOWorld**: 모델 파일명에 "-world" 포함 시 활성화, 오픈 보캐뷸러리 탐지
- **YOLOE**: 모델 파일명에 "yoloe" 포함 시 활성화, 별도 세그멘테이션 학습 지원

## 하위 디렉토리 (각각 train.py, val.py, predict.py 포함)
- `detect/` — 객체 탐지
- `segment/` — 인스턴스 세그멘테이션
- `classify/` — 이미지 분류
- `pose/` — 포즈 추정
- `obb/` — 회전 바운딩 박스 탐지
- `world/` — YOLOWorld 학습 (train.py, train_world.py)
- `yoloe/` — YOLOE 학습 (train.py, train_seg.py, predict.py, val.py)
