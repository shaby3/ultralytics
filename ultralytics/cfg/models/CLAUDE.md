# cfg/models/ - 모델 아키텍처 설정

## 개요
모델 아키텍처(backbone + head)를 스케일링 변형과 함께 정의하는 YAML 파일. 각 버전 서브디렉토리에 태스크별 YAML 파일 존재.

## Model YAML 형식
```yaml
nc: 80
scales:
  n: [0.50, 0.25, 1024]   # [depth_multiple, width_multiple, max_channels]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # [from, repeats, module, args]
head:
  - [-1, 1, Detect, [nc]]
```

## 새 모델 변형 추가 방법
1. `cfg/models/{version}/`에 backbone + head 레이어를 `[from, repeats, module, args]` 형식으로 정의하는 YAML 생성
2. `nn/modules/`의 기존 모듈 사용 또는 새 모듈 추가
3. 표준 태스크 헤드 사용 시 자동 인식됨 — 별도 등록 불필요

## 네이밍 컨벤션
- `{model}{version}{scale}.yaml` — 기본 탐지 (예: yolo26n.yaml)
- `{model}{version}{scale}-{task}.yaml` — 태스크 변형 (예: yolo26n-seg.yaml, yolo26n-pose.yaml)
- `yoloe-{version}.yaml` / `yoloe-{version}-seg.yaml` — YOLOE 변형
- 특수: yolov8-world.yaml, yolov8-rtdetr.yaml

## 버전 디렉토리
- `26/` — YOLO26 (최신, end2end, reg_max:1)
- `12/` — YOLO12
- `11/` — YOLO11
- `v10/` — YOLOv10
- `v9/` — YOLOv9
- `v8/` — YOLOv8 (world, rtdetr 설정 포함)
- `v6/` — YOLOv6
- `v5/` — YOLOv5
- `v3/` — YOLOv3
- `rt-detr/` — RT-DETR 모델

## YAML → 모델 변환 흐름
`nn/tasks.py` → `yaml_model_load()`가 YAML 파싱 → 모듈명을 `nn/modules/` 클래스로 해석 → `torch.nn.Sequential` 구축. `guess_model_task()`가 헤드 타입으로 태스크 추론.
