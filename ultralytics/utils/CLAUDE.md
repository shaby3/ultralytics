# utils/ - 공유 유틸리티

## 개요
코드베이스 전반에서 사용되는 유틸리티. 손실 함수, 메트릭, 텐서 연산, 시각화, PyTorch 헬퍼, 통합 콜백 포함.

## 핵심 파일
- `__init__.py` — 글로벌 상수 (ROOT, DEFAULT_CFG, SETTINGS, RANK, LOGGER, ASSETS 등), IterableSimpleNamespace, YAML 헬퍼, colorstr, 환경 감지
- `loss.py` — 모든 손실 함수: v8DetectionLoss, v8SegmentationLoss, v8ClassificationLoss, v8PoseLoss, v8OBBLoss, E2ELoss, VarifocalLoss, BboxLoss, v26 변형 (PoseLoss26 등)
- `metrics.py` — 평가 메트릭: ConfusionMatrix, DetMetrics, SegmentMetrics, PoseMetrics, ClassifyMetrics, OBBMetrics, bbox_iou, ap_per_class
- `ops.py` — 텐서 연산: non_max_suppression, xywh2xyxy, xyxy2xywh, scale_boxes, process_mask, crop_mask, segment2box
- `tal.py` — Task-Aligned Assigner (학습용 라벨 할당): TaskAlignedAssigner, RotatedTaskAlignedAssigner, dist2bbox, make_anchors
- `torch_utils.py` — PyTorch 헬퍼: select_device, ModelEMA, EarlyStopping, init_seeds, fuse_conv_and_bn, model_info, autocast, smart_inference_mode
- `plotting.py` — 시각화: Annotator 클래스, plot_results, plot_images, Colors
- `checks.py` — 환경/의존성 체크: check_imgsz, check_file, check_requirements
- `nms.py` — Non-maximum suppression 구현
- `benchmarks.py` — 모델 벤치마킹 유틸리티
- `downloads.py` — 파일 다운로드 헬퍼
- `autobatch.py` — 자동 배치 사이즈 선택
- `autodevice.py` — 자동 디바이스 선택
- `instance.py` — Instances 클래스 (바운딩 박스 조작)

## 하위 디렉토리
- `callbacks/` — 학습/예측 라이프사이클 콜백 (@callbacks/CLAUDE.md)
- `export/` — 내보내기 유틸리티 (@export/CLAUDE.md)
