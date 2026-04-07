# cfg/ - 설정 시스템

## 개요
기본 하이퍼파라미터, 모델 아키텍처, 데이터셋 정의, 트래커 설정 등 모든 설정을 관리. CLI의 핵심.

## Config 병합 순서 (나중이 우선)
`cfg/default.yaml` → task defaults → model YAML → CLI args → programmatic overrides

## 핵심 파일
- `__init__.py` — 핵심 설정 로직: `get_cfg()`, `cfg2dict()`, `entrypoint()`, `handle_yolo_hub()`, `handle_yolo_settings()`. 타입 검증: `CFG_FLOAT_KEYS`, `CFG_INT_KEYS`, `CFG_BOOL_KEYS`. 상수: `MODES`, `TASKS`, `TASK2DATA`, `TASK2METRIC`, `SOLUTION_MAP`.
- `default.yaml` — 학습/검증/예측/내보내기 전체 기본 설정값.

## 패턴
- 모든 CLI 라우팅은 `__init__.py`의 `entrypoint()`를 통해 처리
- `SOLUTION_MAP`이 솔루션명을 클래스에 매핑
- `TASKS = frozenset({"detect", "segment", "classify", "pose", "obb"})`
- `MODES = frozenset({"train", "val", "predict", "export", "track", "benchmark"})`

## 하위 디렉토리
- `models/` — 모델 아키텍처 YAML 설정 (@models/CLAUDE.md)
- `datasets/` — 데이터셋 YAML 설정 (coco.yaml, VOC.yaml 등)
- `trackers/` — 트래커 설정 YAML (botsort.yaml, bytetrack.yaml)
