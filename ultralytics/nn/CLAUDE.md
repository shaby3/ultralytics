# nn/ - 신경망 모듈

## 개요
모든 신경망 빌딩 블록, 모델 파싱, 멀티 백엔드 추론, 모델 구축 파이프라인. YAML 설정이 실행 가능한 PyTorch 모델로 변환되는 곳.

## 모델 정의 흐름
1. YAML 설정 (`cfg/models/{version}/`)이 backbone + head 레이어를 스케일링 변형과 함께 정의
2. `tasks.py`의 `yaml_model_load()`가 YAML을 파싱하고, `nn/modules/`의 클래스로 모듈명을 해석하여 `torch.nn.Sequential` 구축
3. `guess_model_task()`가 헤드 타입, 모델 접미사, 또는 체크포인트 메타데이터로 태스크 추론

## 핵심 파일
- `tasks.py` — 중앙 모델 빌더: `yaml_model_load()`, `parse_model()`, `guess_model_task()`. DetectionModel, SegmentationModel, PoseModel, OBBModel, ClassificationModel, WorldModel, YOLOEModel, YOLOESegModel 정의. 모두 BaseModel 상속.
- `autobackend.py` — AutoBackend 클래스: PyTorch, ONNX, TensorRT, TFLite, OpenVINO, CoreML 등 통합 추론. `check_class_names()`.
- `text_model.py` — 오픈 보캐뷸러리 모델용 텍스트 인코더 (CLIP 통합)

## 하위 디렉토리
- `modules/` — 모든 NN 빌딩 블록 (@modules/CLAUDE.md)
- `backends/` — 백엔드별 추론 어댑터 (@backends/CLAUDE.md)
