# ultralytics/ - 소스 패키지

## 개요
Ultralytics YOLO 프레임워크의 루트 패키지. `__init__.py`에서 lazy import를 통해 모델 클래스(YOLO, YOLOWorld, YOLOE, SAM, FastSAM, NAS, RTDETR)를 export.

## 학습 파이프라인 (end-to-end 흐름)
`Model.train()` → 태스크별 Trainer → `BaseTrainer` (`engine/trainer.py`) 에폭 루프 실행:
- 데이터 로딩: `data/build.py` + `data/dataset.py`
- 증강: `data/augment.py` (mosaic, mixup, cutmix, copy-paste)
- 손실 함수: `utils/loss.py`
- 메트릭: `utils/metrics.py`
- 콜백: `utils/callbacks/`
- 체크포인팅, EMA, 워밍업, 코사인 LR 스케줄링

## 주요 모듈 위치 (cross-reference)
- **Engine** (기본 클래스): `engine/model.py`, `engine/trainer.py`, `engine/predictor.py`, `engine/validator.py`, `engine/exporter.py`, `engine/results.py`
- **Export**: `engine/exporter.py` (PyTorch → ONNX/TF/TFLite/OpenVINO/CoreML 등)
- **Results**: `engine/results.py` (예측 컨테이너: `show()`, `save()`, `plot()`)
- **Optimizer**: `optim/` (MuSGD 변형)

## 하위 디렉토리
- `cfg/` — 설정 시스템 (@cfg/CLAUDE.md)
- `data/` — 데이터 로딩 및 증강 (@data/CLAUDE.md)
- `hub/` — Ultralytics HUB 통합 (@hub/CLAUDE.md)
- `models/` — 모델 구현 (@models/CLAUDE.md)
- `nn/` — 신경망 빌딩 블록 (@nn/CLAUDE.md)
- `solutions/` — 사전 구축 CV 솔루션 (@solutions/CLAUDE.md)
- `trackers/` — 객체 트래킹 (@trackers/CLAUDE.md)
- `utils/` — 공유 유틸리티 (@utils/CLAUDE.md)
- `engine/` — 기본 엔진 클래스 (trainer, predictor, validator, exporter, model, results)
- `optim/` — 커스텀 옵티마이저
- `assets/` — 테스트용 샘플 이미지
