# nn/backends/ - 추론 백엔드 어댑터

## 개요
내보낸 모델의 추론을 위한 백엔드별 어댑터. `nn/autobackend.py`의 AutoBackend가 모델 파일 확장자에 따라 적절한 백엔드를 자동 선택하여 통일된 `predict()` 인터페이스를 제공.

## 핵심 파일
- `base.py` — BaseBackend 기본 클래스 (모든 백엔드가 상속)
- `pytorch.py` — 네이티브 PyTorch 백엔드
- `onnx.py` — ONNX Runtime 백엔드
- `tensorrt.py` — TensorRT 백엔드
- `openvino.py` — OpenVINO 백엔드
- `coreml.py` — CoreML 백엔드 (macOS/iOS)
- `tensorflow.py` — TensorFlow/TFLite/TF.js 백엔드
- `executorch.py` — ExecuTorch (모바일) 백엔드
- `paddle.py` — PaddlePaddle 백엔드
- `ncnn.py` — NCNN (모바일) 백엔드
- `mnn.py` — MNN 백엔드
- `rknn.py` — RKNN (Rockchip NPU) 백엔드
- `triton.py` — Triton Inference Server 백엔드
- `axelera.py` — Axelera AI 백엔드

## 패턴
- 각 백엔드는 BaseBackend를 상속하여 일관된 인터페이스 구현
- AutoBackend가 모델 파일 확장자 기반으로 올바른 백엔드 자동 선택
