# YOLOv8 Knowledge Distillation on Pascal VOC

YOLOv8 기반 **Knowledge Distillation(KD)** 연구용 ultralytics 포크.
Detect head / Neck 의 중간 feature 를 증류해 소형 모델(student)의 성능을 끌어올리는 것이 목표다.

> 원본 ultralytics README 는 [README_BACKUP.md](README_BACKUP.md) 에 보존되어 있다.

---

## 1. 실험 구성

| 역할 | 모델 | 비고 |
|------|------|------|
| Student | **YOLOv8n** | 증류 대상 |
| Teacher A | **YOLOv8s** | VOC 학습본 / COCO-pretrained 두 버전 모두 사용 |
| Teacher B | **YOLOv8m** | VOC 학습본 / COCO-pretrained 두 버전 모두 사용 |

데이터셋은 **Pascal VOC** (`ultralytics/cfg/datasets/VOC.yaml`) 고정.

- 클래스 20개
- train 16,551장 (train2007 + val2007 + train2012 + val2012)
- val 4,952장 (test2007)
- 로컬 경로: `C:\Users\SSAFY\datasets\VOC` (약 5.5GB)

### 진행 순서

1. **Baseline 확보** — yolov8n / s / m 을 VOC 로 각각 학습 (n 은 student baseline, s·m 은 baseline 지표 겸 VOC-teacher)
2. **KD 본실험** — teacher 4종(s/m × VOC학습/COCO-pretrained) 조합으로 증류
3. 결과를 아래 [5. 실험 결과](#5-실험-결과) 표에 누적

---

## 2. 환경

| 항목 | 값 |
|------|-----|
| GPU | RTX 4050 Laptop **6GB** |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| Python | 3.11.15 (`.venv`, uv 관리) |
| ultralytics | 8.4.34 (editable install) |
| 실험 추적 | **W&B** (`wandb` 0.28.1, 활성화됨) |

`.venv` 는 uv 로 생성되어 **pip 이 없다.** 패키지 설치는 uv 를 쓴다.

```bash
~/.local/bin/uv.exe pip install --python ./.venv/Scripts/python.exe <package>
```

W&B 는 로그인된 상태이며, 끄고 싶을 때는 `WANDB_MODE=disabled` 를 앞에 붙인다.

---

## 3. 학습 설정 (확정)

세 baseline 공통:

| 항목 | 값 |
|------|-----|
| epochs | **100** |
| patience | **30** (early stopping) |
| imgsz | 640 |
| amp | True |
| optimizer | `auto` → **MuSGD** (아래 주의사항 참조) |
| device | 0 |
| workers | 2 |

배치는 **1에폭 실측(검증 포함)으로 확정**했다. AutoBatch 추천값은 과도하게 보수적이어서 사용하지 않았다.

| 모델 | batch | val batch | peak VRAM | 1 epoch | 100 epoch (추정) |
|------|:---:|:---:|:---:|:---:|:---:|
| yolov8n | **32** | 64 | 3.68GB / 6GB | 8.9분 | 약 14.8시간 |
| yolov8s | **16** | 32 | 3.36GB / 6GB | 10.8분 | 약 17.9시간 |
| yolov8m | **8** | 16 | 3.20GB / 6GB | 22.7분 | 약 37.8시간 |

(AutoBatch 추천은 각 12 / 4 / 2 였다.)

---

## 4. 실행 방법

### Baseline 학습

순차로 실행한다. 동시에 돌리면 VRAM 부족으로 실패한다.

```bash
.venv/Scripts/python.exe scripts/voc/train/train_yolov8n_voc.py
```

```bash
.venv/Scripts/python.exe scripts/voc/train/train_yolov8s_voc.py
```

```bash
.venv/Scripts/python.exe scripts/voc/train/train_yolov8m_voc.py
```

결과는 `runs/detect/voc_baseline_yolov8{n,s,m}/` 에 저장된다.

### KD 학습

| 스크립트 | KD 위치 | aligner | distill config |
|---------|---------|---------|----------------|
| `scripts/run_phase6_kd.py` | Detect head | ConvAligner | `distill_cfg.yaml` |
| `scripts/run_phase6_kd_convbn.py` | Detect head | ConvBNAligner | `distill_cfg_convbn.yaml` |
| `scripts/run_phase6_kd_convbnsilu.py` | Detect head | **ConvBNSiLUAligner** | `distill_cfg_convbnsilu.yaml` |
| `scripts/run_neck_kd.py` | Neck 출력 | ConvAligner | `distill_neck_cfg.yaml` |

> 이 스크립트들은 이전 실험 세팅(epochs 50, batch 32)을 담고 있다.
> 본실험 전에 위 [3. 학습 설정](#3-학습-설정-확정)에 맞춰 갱신해야 한다.

### 증류 지점

**Head KD** — Detect head(`model.22`) 내부, box(cv2)/cls(cv3) 분기의 **2번째 conv 출력** × 3 스케일 = 6 지점

```
model.22.cv2.{0,1,2}.1    # box 분기
model.22.cv3.{0,1,2}.1    # cls 분기
```

**Neck KD** — neck 출력 P3/P4/P5 = 3 지점 (layer `15`, `18`, `21`)

loss 는 MSE, weight 1.0.

### Aligner 는 ConvBNSiLU 로 고정

student feature 를 teacher feature 에 정렬하는 모듈. **ConvBNSiLUAligner 로 고정**한다.

근거: 증류 대상인 teacher feature 는 `Conv(Conv2d+BN+SiLU)` 블록의 출력이다
([`head.py`](ultralytics/nn/modules/head.py) 의 `cv2`/`cv3` = `Sequential(Conv, Conv, nn.Conv2d)`, 증류 지점은 인덱스 `.1`).
따라서 aligner 도 BN+SiLU 로 끝나야 동일한 분포 공간에 놓이고, MSE loss 가 분포 불일치를 오차로 오인하지 않는다.
YOLOv8 이 모든 conv 를 Conv2d+BN+SiLU 로 구성한다는 아키텍처 일관성과도 맞는다.

| aligner | 구조 | teacher 분포 정합 |
|---------|------|:---:|
| ConvAligner | Conv2d → ReLU → Conv2d (선형 출력) | ✗ |
| ConvBNAligner | Conv+BN+SiLU → Conv2d (선형 출력) | ✗ |
| **ConvBNSiLUAligner** | Conv+BN+SiLU → Conv+BN+SiLU | **✓** |

---

## 5. 실험 결과

아직 없음. baseline 학습 완료 후 아래 표에 채운다.

### Baseline

| 모델 | batch | epochs | mAP50 | mAP50-95 | Precision | Recall |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| yolov8n | 32 | 100 | - | - | - | - |
| yolov8s | 16 | 100 | - | - | - | - |
| yolov8m | 8 | 100 | - | - | - | - |

### KD

| 실험 | teacher | KD 위치 | aligner | mAP50 | mAP50-95 | ΔmAP50-95 |
|------|---------|---------|---------|:---:|:---:|:---:|
| - | - | - | - | - | - | - |

---

## 6. 주의사항 (중요)

이 포크에서 실제로 겪은 함정들. 새 실험 전에 반드시 확인한다.

### 6.1 C2f 는 upstream 원본 상태를 유지할 것

과거 구조적 프루닝을 위해 `C2f` 의 `cv1`(2c 채널)을 `cv0`+`cv1` 두 conv 로 분리한 패치가 있었다.
이 상태에서는 COCO 사전학습 체크포인트의 키가 매칭되지 않아 **가중치가 조용히 유실된다.**

| 상태 | yolov8n 가중치 전이 |
|------|--------------------|
| 프루닝 패치 적용 | `Transferred 315/403` ← 88개 유실 |
| **upstream 원복 (현재)** | `Transferred 355/355` ✓ |

C2f 는 backbone/neck 의 핵심 블록이라, 유실 상태로 학습하면 사전학습 효과를 크게 잃는다.
프루닝이 필요해지면 [docs_internal/C2F_PRUNING_PATCH.md](docs_internal/C2F_PRUNING_PATCH.md) 의 재적용 절차를 따르고,
`convert_weights.py` 로 체크포인트를 먼저 변환해야 한다.

### 6.2 VOC 에서 `Transferred 319/355` 는 정상

VOC 는 nc=20 이라 Detect head 의 cls 분기(`cv3`) 채널 수가 COCO(nc=80)와 달라진다.
해당 36개 항목은 새로 초기화되는 것이 정상이다. backbone·neck 은 전량 전이된다.
학습 로그에서 이 숫자를 항상 확인해 백본까지 유실되지 않았는지 점검한다.

### 6.3 `optimizer=auto` 는 100에폭에서 MuSGD 를 고른다

[`trainer.py`](ultralytics/engine/trainer.py) 의 분기:

```python
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

100에폭이면 iterations 가 5만~20만이라 **항상 MuSGD** 다(1~2에폭 테스트에서는 AdamW 가 선택되어 로그가 달라 보인다).
또한 `auto` 는 `lr0` / `momentum` 설정을 **무시**하고 MuSGD 기본값을 쓴다.
baseline 과 KD 실험이 모두 같은 조건을 타므로 비교에는 문제가 없다.

### 6.4 AutoBatch 는 과도하게 보수적

최악의 객체 수를 가정해 추정하므로 실측보다 훨씬 작은 값을 낸다(추천 12/4/2 vs 실측 32/16/8).
배치는 짧은 실측으로 정하는 편이 낫다. 다만 트레이너 밖에서 `autobatch()` 를 직접 호출하면
데이터셋 컨텍스트가 없어 프로파일링이 실패한다.

### 6.5 `project=` 인자는 쓰지 말 것

[`cfg/__init__.py`](ultralytics/cfg/__init__.py) 는 상대 경로 project 를 `RUNS_DIR/task/project` 로 합친다.
`project="runs/detect"` 를 주면 `runs/detect/runs/detect/<name>` 으로 중첩된다.
`name` 만 지정하면 `runs/detect/<name>` 에 올바르게 저장된다.

### 6.6 val batch 는 train batch 의 2배 (하드코딩)

`trainer.py` 에서 `batch_size * 2` 로 고정되어 있고 별도 인자가 없다.
검증은 추론이라 메모리를 덜 쓰지만, 배치를 키울 때는 이 2배를 감안한다.
따로 조절하려면 `DetectionTrainer` 를 상속해 `get_dataloader` 에서 `mode == "val"` 일 때 배치를 바꾸고
`train(trainer=...)` 로 넘긴다.

### 6.7 Windows 에서는 `if __name__ == "__main__":` 필수

`workers > 0` 이면 DataLoader 가 spawn 으로 자식 프로세스를 만든다(Windows 는 fork 없음).
가드가 없으면 자식이 스크립트 전체를 재실행해 **프로세스가 무한 증식**한다.
학습 호출은 반드시 가드 안에 둔다.
