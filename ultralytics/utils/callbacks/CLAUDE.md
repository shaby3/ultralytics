# utils/callbacks/ - 라이프사이클 콜백

## 개요
학습, 검증, 예측, 내보내기 이벤트에 훅을 거는 콜백 시스템. 외부 로깅 플랫폼과 통합.

## 핵심 파일
- `base.py` — 기본 콜백 정의. 모든 콜백 훅: on_train_start, on_train_epoch_end, on_val_start, on_predict_start, on_export_start 등. `get_default_callbacks()`로 전체 콜백 딕셔너리 반환.
- `__init__.py` — 콜백 레지스트리 및 관리

## 로거 통합 (각 플랫폼별 콜백 자동 등록)
- `tensorboard.py` — TensorBoard 로깅
- `wb.py` — Weights & Biases (wandb) 로깅
- `mlflow.py` — MLflow 로깅
- `comet.py` — Comet ML 로깅
- `clearml.py` — ClearML 로깅
- `neptune.py` — Neptune.ai 로깅
- `dvc.py` — DVC (Data Version Control) 로깅
- `raytune.py` — Ray Tune 하이퍼파라미터 최적화 콜백
- `hub.py` — Ultralytics HUB 콜백 (진행 상황 보고)
- `platform.py` — 플랫폼별 콜백

## 패턴
- 콜백은 이벤트명 → callable 리스트의 딕셔너리 구조
- Trainer/Predictor/Validator가 `self.run_callbacks(event_name)`으로 각 라이프사이클 포인트에서 호출
- 해당 패키지가 감지되면 통합 콜백이 자동 등록됨
