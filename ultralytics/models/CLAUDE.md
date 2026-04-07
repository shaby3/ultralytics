# models/ - 모델 구현

## 개요
모든 모델 아키텍처를 보유. 각 모델 패밀리는 자체 서브디렉토리에 model.py (공개 Model 클래스)와 태스크별 train/val/predict 파일을 포함. 모든 모델 클래스는 `ultralytics/__init__.py`를 통해 최상위 import 가능.

## 공개 모델 클래스
- **YOLO**, **YOLOWorld**, **YOLOE** (models/yolo/)
- **SAM** (models/sam/)
- **FastSAM** (models/fastsam/)
- **NAS** (models/nas/)
- **RTDETR** (models/rtdetr/)

## 패턴
- 각 모델 디렉토리에 `model.py`가 `engine.model.Model`을 상속하는 클래스 포함
- Model 기본 클래스 (`engine/model.py`)가 처리: `_new` (YAML에서), `_load` (.pt에서), predict, train, val, export, benchmark
- `task_map` 프로퍼티로 태스크 라우팅 (태스크 문자열 → trainer/validator/predictor 클래스 매핑)

## 하위 디렉토리
- `yolo/` — YOLO 패밀리 모델 (@yolo/CLAUDE.md)
- `sam/` — Segment Anything Model (@sam/CLAUDE.md)
- `fastsam/` — FastSAM 모델
- `nas/` — Neural Architecture Search 모델
- `rtdetr/` — RT-DETR 실시간 트랜스포머 디텍터
- `utils/` — 공유 모델 유틸리티 (loss.py, ops.py)
