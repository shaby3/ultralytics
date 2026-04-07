# utils/export/ - 내보내기 유틸리티

## 개요
PyTorch 모델을 다양한 배포 포맷으로 내보내기 위한 헬퍼 모듈. 각 파일은 `engine/exporter.py`에서 사용되는 포맷별 전처리, 변환, 검증 로직을 제공.

## 핵심 파일
- `engine.py` — 코어 내보내기 엔진 유틸리티
- `openvino.py` — OpenVINO 내보내기 헬퍼
- `coreml.py` — CoreML 내보내기 헬퍼
- `tensorflow.py` — TensorFlow/TFLite 내보내기 헬퍼
- `torchscript.py` — TorchScript 내보내기 헬퍼
- `executorch.py` — ExecuTorch 모바일 내보내기 헬퍼
- `paddle.py` — PaddlePaddle 내보내기 헬퍼
- `ncnn.py` — NCNN 모바일 내보내기 헬퍼
- `mnn.py` — MNN 내보내기 헬퍼
- `rknn.py` — RKNN (Rockchip NPU) 내보내기 헬퍼
- `axelera.py` — Axelera AI 내보내기 헬퍼
- `imx.py` — i.MX NXP 내보내기 헬퍼

## 패턴
- 각 모듈은 포맷별 변환/양자화 로직 제공
- `engine/exporter.py`의 내보내기 파이프라인에서 호출됨
- `nn/backends/`는 내보낸 모델의 추론을 담당 (상호 보완 관계)
