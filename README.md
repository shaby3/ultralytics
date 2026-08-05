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

**Baseline 은 확보 완료** — yolov8n / s / m 을 VOC 로 각각 학습했다. 지표는 [6. 실험 결과](#6-실험-결과) 참조.
n 은 student baseline, s·m 은 baseline 지표 겸 VOC-teacher 로 쓴다.

KD 본실험은 축이 3개다: **증류 위치**(3수준) × **teacher 출처**(VOC학습/COCO-pretrained) × **teacher 크기**(s/m).
전수조사는 12런이고 GPU 가 1주일 넘게 묶인다. 그래서 **한 phase 에서 한 축만 바꾸고, 이긴 설정을 다음 phase 로 넘긴다.**

| Phase | 런 | 바뀌는 축 | 답하는 질문 |
|:---:|------|-----------|-------------|
| **0** | — (1에폭 측정만) | — | batch 확정, 위치별 `kd_loss` 크기 |
| **1** | neck / head`.0` / head`.1` × **s-VOC** | 증류 위치 (3수준) | 어디서 증류하는 게 좋은가 |
| **2** | Phase 1 승자 위치 × **s-COCO** | teacher 출처 | VOC 로 학습한 teacher 가 나은가 |
| **3** | Phase 1 승자 위치 × **m**(Phase 2 승자 출처) | teacher 크기 | teacher 를 키우면 나아지는가 |

**총 5런.** 위치가 3수준으로 가장 넓으니 가장 싼 teacher(s)로 먼저 쓸어야 총비용이 최소가 된다.
그리고 위치가 나쁘면 gain 이 0 에 가까워서, 위치를 먼저 정하지 않으면 teacher 비교가 전부 노이즈 위에서 이뤄진다.

### 각 phase 가 허용하는 주장의 범위

**Q3(teacher 크기)은 같은 출처의 s 와 비교한다.** Phase 3 의 m 은 Phase 2 승자 출처를 따르므로,
비교 대상은 Phase 2 가 COCO 로 이겼으면 s-COCO(Phase 2), VOC 로 이겼으면 s-VOC(Phase 1) 다.
어느 쪽이든 **한 축만 다른 비교**가 되고, 그 런은 이미 손에 있다.

teacher 크기 추이는 3점으로 읽는다: **teacher 없음(baseline n) → s → m.**
teacher 가 클수록 나빠지는 구간이 보이면 capacity gap 을 짚는 근거가 된다.

teacher 2×2 그리드(출처 × 크기)에서 **3칸이 채워지고 1칸이 빈다.** 그 칸을 채울지는 Phase 2 결과로 정한다.

| Phase 2 승자 | Phase 3 | 빠지는 칸 |
|:---:|:---:|:---:|
| VOC | m-VOC | m-COCO |
| COCO | m-COCO | m-VOC |

- Phase 2 의 COCO vs VOC 차이가 **작으면(~0.5pt 미만) 빈 칸은 건너뛴다** — 교호작용을 논할 주효과가 없다.
- **크면 1런 추가한다.** 큰 주효과일 때가 교호작용이 결론을 뒤집을 수 있는 상황이다.
  "VOC 학습 teacher 가 낫다"를 s 에서만 재고 일반화하면, m 에서 방향이 뒤집힐 때 논지가 무너진다.

**Phase 1 의 위치 순위는 s-VOC 에서만 확인한 것이다.** 다른 teacher 에서도 같은 순위인지는 검증하지 않는다
(greedy 탐색의 한계). 필요하면 2위 위치를 최고 teacher 로 1런 돌려 방어한다.

### 결과 해석 시 주의

**단일 seed 로 0.3pt 차이를 순위로 주장하지 않는다.** Phase 1 의 세 위치가 0.3pt 안에 몰리면
"구분되지 않음"으로 보고하거나 상위 2개만 seed 를 바꿔 반복한다.
(`val/` 자체는 결정적이다 — [§7.7](#77-val-의-rect-기본값은-true-다--defaultyaml-을-믿으면-안-된다) 참조.
학습 seed 노이즈와 val 재현성은 다른 문제다.)

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
.venv/Scripts/python.exe scripts/voc/baseline/train_yolov8n.py
```

```bash
.venv/Scripts/python.exe scripts/voc/baseline/train_yolov8s.py
```

```bash
.venv/Scripts/python.exe scripts/voc/baseline/train_yolov8m.py
```

세 개를 이어서 돌리려면 한 줄로 묶는다. `;` 이므로 하나가 실패해도 다음이 진행된다
(앞이 성공해야만 다음으로 넘기려면 `&&`). Windows 에서 화면보호기·화면 꺼짐은 프로세스를 죽이지 않고,
이 머신은 절전 대기시간이 AC/DC 모두 `0`(=사용 안 함)이라 터미널 창만 열어두면 된다.

```bash
.venv/Scripts/python.exe scripts/voc/baseline/train_yolov8n.py; .venv/Scripts/python.exe scripts/voc/baseline/train_yolov8s.py; .venv/Scripts/python.exe scripts/voc/baseline/train_yolov8m.py
```

각 스크립트는 학습 후 `best.pt` 로 val 재평가까지 수행한다. 결과는
`runs/detect/voc/baseline/yolov8{n,s,m}/{train,val}/` 에 저장된다 — 아래 [5. 결과 저장 구조](#5-결과-저장-구조) 참조.

### KD 학습

**Phase 1 — 증류 위치 3수준.** teacher 는 셋 다 s-VOC 고정, aligner·loss 도 동일하고 `layers` 와 `weight` 만 다르다.

| distill config | KD 위치 | 지점 | student n → teacher s 채널 | weight |
|----------------|---------|:---:|----------------------------|:---:|
| `distill_neck_from_8s_voc.yaml` | neck 출력 (layer 15/18/21) | 3 | 64→128, 128→256, 256→512 | **20** |
| `distill_head0_from_8s_voc.yaml` | Detect head **1번째** conv | 6 | box 64→64 ×3, cls 64→128 ×3 | **10** |
| `distill_head1_from_8s_voc.yaml` | Detect head **2번째** conv | 6 | box 64→64 ×3, cls 64→128 ×3 | **1** (기준) |

### weight 는 위치별 kd_loss 실측으로 정규화했다

`weight=1.0` 을 셋에 공통으로 주면 위치 비교가 성립하지 않는다. kd_loss 는 MSE 라 그 지점 feature 의
분산에 비례하는데, 위치마다 스케일이 다르기 때문이다. 1에폭 실측 (teacher s-VOC, batch 32):

| 위치 | kd_loss (1에폭 평균) | task loss 대비 초기 KD 비중 | head1 대비 |
|------|:---:|:---:|:---:|
| neck | 0.318 | 6.4% | 1/18.8 |
| head`.0` | 0.667 | 12.6% | 1/9.0 |
| head`.1` | 5.992 | 56.3% | 1 |

**19배 차이다.** 이대로 돌리면 head1 은 KD 가 학습 신호의 절반이고 neck 은 6% — 위치 비교가 아니라
KD 강도 비교가 된다. 그래서 **head1 을 1.0 기준으로 두고**(이전 회차 실험과 이어진다) 나머지를
역수로 올려 초기 KD 기여를 맞췄다: neck 18.8→**20**, head0 9.0→**10** (반올림).

teacher 를 s→m 으로 바꿔도 neck kd_loss 는 0.318→0.330 으로 거의 안 변한다.
**크기를 지배하는 건 teacher 가 아니라 위치다** — Phase 2·3 에서 weight 를 재보정하지 않는 근거.

한계: 이 정규화는 1에폭 시점의 기여율만 맞춘다. 학습이 진행되면 비율은 다시 갈린다.
위치별 최적 weight 스윕이 엄밀하지만 런이 3배가 되어, 자릿수 교락만 막는 선에서 멈춘 것이다.

### KD 학습 비용 (Phase 0, 1에폭 실측)

> 처음 Phase 0 프로브에서 검증 구간 peak 가 7.5~10.2GB 로 물리 VRAM(6.1GB)을 넘어
> 시스템 RAM 스필 + 시스템 전체 렉이 발생했다. 원인은 activation 이 아니라
> **distiller 의 EMA 유령 hook 메모리 리크**였다 — §7.8 참조. 수정 후 같은 조합(head1)의
> peak 가 10.2G → **2.9G** 로, 에폭 시간이 12.4분 → **7.8분**으로 내려왔다.

**batch 는 5런 전부 16 으로 고정한다.** 32 도 돌지만 학습 구간이 5.8G/6.1G 로 빠듯해서
여유를 둔 선택이다 (batch 16 학습 구간은 ~2.6G).

**baseline(batch 32)과의 비교는 유지된다** — ultralytics 는 gradient accumulation 으로
명목 batch(`nbs=64`)를 맞춘다. 32 는 2번, 16 은 4번 누적이라 optimizer 가 보는 유효 batch 는
둘 다 64 이고 weight_decay 스케일도 같다. 남는 차이는 BN 통계(배치당 16 vs 32)뿐이다.
KD 5런끼리는 전부 batch 16 이라 내부 비교(Q1·Q2·Q3)에는 아예 영향이 없고,
baseline 대비 절대 gain 에만 이 경미한 차이가 얹힌다.

리크 수정 후 실측(batch 16, 검증 batch*2=32): **head1 7.8분/에폭 → 100에폭 약 13h.**
head0 은 head1 과 동급, neck 은 KD 지점이 적어 이보다 빠르다(11~13h 추정).
baseline n(4.74분/에폭) 대비 약 1.6배. **Phase 1 = 약 1.6일, 5런 총 2.5~3일** 수준으로 잡는다.

kd_loss 는 배치 평균 MSE 라 batch 크기와 무관하다 — 위 weight 정규화(batch 32 실측)는 그대로
유효하고, 리크 수정 후 head1 재실측(5.79)도 이전 값(5.99)과 일치한다.
보고용 `val/` 재평가(`VAL_ARGS`)는 batch 32 를 유지한다 — baseline n 의 `val/` 과 같은 조건이고,
이 단계는 teacher 가 없어 가볍다.

head0 과 head1 은 **채널이 완전히 같다** — `cv2`/`cv3` 가 `Sequential(Conv, Conv, nn.Conv2d)` 라
`.0` 과 `.1` 의 출력 채널이 동일하기 때문이다. aligner 파라미터 수까지 같아서 세 수준 중 가장 깨끗한 비교다.

neck 은 지점이 3개뿐이지만 채널이 커서 **aligner 파라미터는 head 보다 훨씬 많다.**
neck 이 이기면 "위치 효과"인지 "aligner 용량 효과"인지 해석에 단서를 달아야 한다.

> 세 config 는 서로 대응이 맞아야 비교가 성립한다. teacher·aligner·loss 를 한쪽만 바꾸면
> Q2 가 위치 × 그 항목의 교락이 된다. weight 는 예외로 **일부러 다르다** — 위 정규화 참조.

Phase 1 실행 — baseline 과 마찬가지로 순차 실행한다 (동시 실행은 VRAM 부족):

```bash
.venv/Scripts/python.exe scripts/voc/kd_neck/train_yolov8n_from_8s.py; .venv/Scripts/python.exe scripts/voc/kd_head0/train_yolov8n_from_8s.py; .venv/Scripts/python.exe scripts/voc/kd_head1/train_yolov8n_from_8s.py
```

각 스크립트는 baseline 스크립트와 같은 골격이다 — 학습 후 `best.pt` 재평가(`val/`)와
지표 기록(`results.csv` · `metrics.json`)까지 수행한다.

Phase 2·3 은 승자 위치의 config 와 스크립트를 복사해 `teacher.model` 만 바꾼다
(`_from_8s_coco` / `_from_8m`). 결과 경로 규칙은 [5. 결과 저장 구조](#5-결과-저장-구조) 참조.

> 이전 회차(epochs 50)의 `run_phase6_kd*.py` · `run_neck_kd.py` 와 그 config 들은 삭제했다.
> 필요하면 git 히스토리에서 복구한다.

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

## 5. 결과 저장 구조

실험 하나 = 디렉토리 하나. **경로가 축(dataset / method / variant)을 담고, `name` 은 stage 만 담는다.**

```
runs/detect/
  voc/                        # dataset
    baseline/                 # method
      yolov8n/                # variant ← 실험의 원자 단위
        train/                #   학습. results.csv = 에폭별 지표, args.yaml = 하이퍼파라미터
        val/                  #   학습 후 best.pt 재평가. results.csv = 클래스별, metrics.json = 스칼라
      yolov8s/
      yolov8m/
    kd_head1/                 # method = 증류 위치
      yolov8n_from_8s/        #   Phase 1 — teacher s-VOC
      yolov8n_from_8s_coco/   #   Phase 2 — teacher 출처만 교체
      yolov8n_from_8m/        #   Phase 3 — teacher 크기만 교체
        train/  val/
    kd_head0/
    kd_neck/
  neu_det/
    baseline/
      yolov8n/
        train/  val/  test/   # neu_det 은 val 과 test 가 분리돼 있다
```

스크립트에서는 이렇게 지정한다:

```python
PROJECT = "voc/baseline/yolov8n"   # runs/detect 는 붙이지 않는다 — 아래 7.5 참조
model.train(project=PROJECT, name="train", ...)
YOLO(model.trainer.best).val(project=PROJECT, name="val", ...)
```

### 네이밍 규칙

| 레벨 | 값 | 금지 |
|------|-----|------|
| dataset | `voc`, `neu_det`, `gc10_det` | — |
| method | `baseline`, `kd_neck`, `kd_head0`, `kd_head1`, `prune_l1` | 실험 회차·phase 번호 (`phase6` 는 시간이 지나면 의미를 잃는다) |
| variant | 모델 크기 + 실제로 갈리는 축 하나. `yolov8n`, `yolov8n_from_8s` | dataset·epochs·stage 중복. `50ep` 은 50/100 을 **둘 다** 돌릴 때만 붙인다 |

`baseline_yolov8n_voc_100epoch` 처럼 쓰지 않는다 — dataset 은 경로에, epochs 는 `train/args.yaml` 에 이미 있다.

### KD 는 method 에 위치, variant 에 teacher

증류 위치가 3수준이라 `kd_head` 하나로는 구분이 안 된다. **위치를 method 로 쪼갠다** —
`kd_neck` / `kd_head0` / `kd_head1`. 그러면 variant 는 teacher 축만 담아 한 축 규칙이 유지된다.

| variant | teacher |
|---------|---------|
| `yolov8n_from_8s` | yolov8s **VOC 학습본** |
| `yolov8n_from_8s_coco` | yolov8s COCO-pretrained |
| `yolov8n_from_8m` | yolov8m VOC 학습본 |
| `yolov8n_from_8m_coco` | yolov8m COCO-pretrained |

**VOC 학습본이 기본이고 COCO 만 표시한다.** VOC 가 이 실험의 주 조건이라 접미사 없는 쪽에 둔다.
`_voc` 를 붙이면 네 variant 중 셋에 붙어 구분에 기여하지 않는다.
(distill config 파일명은 반대로 `_voc` 를 명시한다 — `ultralytics/cfg/` 에는 VOC 아닌 config 도 놓일 수 있다.)

### `val/` 은 왜 따로 두는가

`train/` 안에도 매 에폭 검증 결과가 이미 들어 있다. 별도 `val/` 은 학습이 고른 **`best.pt` 로 다시 한 번 평가**한 것이다.
early stopping 이 걸리면 `last.pt` 와 `best.pt` 가 달라지므로, 보고용 지표는 `val/` 쪽을 쓴다.

### `val/` 지표는 스크립트가 직접 기록한다

ultralytics 는 **`val/` 에 지표를 파일로 남기지 않는다.** curve·confusion matrix PNG 만 저장하고,
`train/` 에 생기는 `results.csv` 도 `args.yaml` 도 만들지 않는다. 숫자가 콘솔에만 찍히고 사라진다.

그래서 스크립트가 `val()` 반환값에서 뽑아 직접 쓴다. 세 파일 다 git 추적 대상이다.

| 파일 | 출처 | 내용 |
|------|------|------|
| `val/results.csv` | `r.to_csv()` | **클래스별** 20행 — Class, Images, Instances, Box-P/R/F1, mAP50, mAP50-95 |
| `val/metrics.json` | `r.results_dict` + `r.speed` + `VAL_ARGS` | 스칼라 지표 4개 + fitness, 추론 속도, **재현용 평가 조건** |

```python
r = YOLO(model.trainer.best).val(project=PROJECT, name="val", exist_ok=True, **VAL_ARGS)
out = Path(r.save_dir)
out.joinpath("results.csv").write_text(r.to_csv(), encoding="utf-8")
out.joinpath("metrics.json").write_text(
    json.dumps({**r.results_dict, "speed": r.speed, "val_args": VAL_ARGS}, indent=2), encoding="utf-8"
)
```

`train/results.csv` 는 에폭별, `val/results.csv` 는 클래스별이다. **파일명이 같지만 스키마가 다르다.**

주의할 점 두 가지:

- **`VAL_ARGS` 는 평가 조건을 명시해 박고 `metrics.json` 에 남긴다.** 기본값에 의존하면 업스트림이 바꿀 때
  과거 결과를 재현할 수 없다. 단 **`rect=True` 다 — `default.yaml` 의 `rect: False` 를 믿으면 안 된다.**
  [§7.7](#77-val-의-rect-기본값은-true-다--defaultyaml-을-믿으면-안-된다) 에 실측값과 함께 적어뒀다.
- **`rect=True` 에서는 `batch` 도 mAP 에 영향을 준다.** rect 는 이미지를 종횡비로 정렬한 뒤
  **배치 단위로** letterbox 모양을 정하므로(`set_rectangle` 의 `bi = floor(arange(ni) / batch_size)`),
  batch 가 바뀌면 묶이는 조합과 패딩이 바뀐다. 영향은 작지만 0 이 아니다 —
  **KD 5런은 val batch 를 32 로 통일한다**(baseline n 과 같은 값이라 비교가 유지된다).
- **`fitness` 는 이 버전에서 mAP50-95 와 같은 값이다.** 가중치가 `[0, 0, 0, 1]` 이다
  ([metrics.py](ultralytics/utils/metrics.py) `Metric.fitness`). 예전 `0.1*mAP50 + 0.9*mAP50-95` 공식이 아니니 별도 지표로 착각하면 안 된다.

**클래스별 지표는 KD 실험의 핵심 자료다.** 전체 mAP 0.5pt 차이만 보면 증류가 무엇을 전달했는지 알 수 없다.
VOC 20클래스 중 어느 클래스가 올랐는지 보면 위치별 KD 의 성격을 논할 근거가 생긴다.

### `test/` 는 조건부

데이터셋 yaml 의 `test` 가 `val` 과 **다른 경로**일 때만 만든다.

| dataset | train/ | val/ | test/ |
|---------|:---:|:---:|:---|
| voc | ○ | ○ | ✗ — `val` 과 `test` 가 둘 다 `images/test2007` 이라 돌려도 같은 숫자가 나온다 |
| neu_det | ○ | ○ | ○ — val→val/test 로 분할했음 |
| gc10_det | ○ | ○ | 확인 필요 |

### 스크립트는 결과 구조를 미러링한다

```
scripts/voc/baseline/train_yolov8n.py         →  runs/detect/voc/baseline/yolov8n/
scripts/voc/kd_head/train_yolov8n_from_8s.py  →  runs/detect/voc/kd_head/yolov8n_from_8s/
```

결과 디렉토리에서 이를 만든 스크립트를, 반대 방향으로도 바로 찾을 수 있다.

### git 추적 범위

`args.yaml` · `results.csv` · `metrics.json` · curve/confusion PNG 는 커밋한다.
`weights/` 와 `wandb/`, `train_batch*.jpg` · `val_batch*.jpg` 는 제외한다 — 배치 이미지는 증강 샘플일 뿐인데 실험당 9장씩 쌓인다.

`.gitignore` 는 `runs/*` 로 통째 무시한 뒤 `runs/detect/` 만 다시 열고 위 확장자를 하나씩 허용하는 구조다.
`*.json` 을 통째로 열지 않고 **`metrics.json` 만 파일명으로 못 박은** 이유는, `save_json=True` 로 val 하면
생기는 `predictions.json`(COCO 포맷 예측 전체)이 딸려 들어오기 때문이다.

---

## 6. 실험 결과

### Baseline

VOC test2007(4,952장) 기준, `val/` 의 **`best.pt` 재평가값**. 세 모델 모두 early stopping 없이 100에폭 완주했다.

| 모델 | batch | epochs | mAP50 | mAP50-95 | Precision | Recall |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| yolov8n | 32 | 100 | 0.8324 | 0.6284 | 0.8148 | 0.7572 |
| yolov8s | 16 | 100 | 0.8628 | 0.6750 | 0.8358 | 0.8085 |
| yolov8m | 8 | 100 | 0.8795 | 0.7077 | 0.8433 | 0.8258 |

student(n) → teacher 간 mAP50-95 격차: **s 기준 +4.66pt / m 기준 +7.93pt.** KD 로 메울 여지가 이만큼이다.

`train/results.csv` 의 마지막 행(ep100)과 비교하면 mAP50-95 차이가 0.0005 미만이다.
early stopping 이 안 걸려 best 에폭이 99 / 99 / 100 이었으니 예상된 결과다.

원본 기록은 `val/metrics.json`(스칼라 + 평가 조건)과 `val/results.csv`(클래스별)에 있다 — 스크립트가 직접 쓴 것이다
([§5 `val/` 지표는 스크립트가 직접 기록한다](#val-지표는-스크립트가-직접-기록한다)).
이 표는 그 값을 옮긴 것이고, KD 실험 결과도 같은 방식으로 누적한다.

| 모델 | 학습 시간 (실측) | §3 추정 | 비고 |
|------|:---:|:---:|------|
| yolov8n | 7.9h | 14.8h | 추정이 크게 보수적이었다 |
| yolov8s | 18.8h | 17.9h | 거의 일치 |
| yolov8m | 33.8h | 37.8h | 추정보다 10% 빠름 |

### KD

| 실험 | teacher | KD 위치 | aligner | mAP50 | mAP50-95 | ΔmAP50-95 |
|------|---------|---------|---------|:---:|:---:|:---:|
| - | - | - | - | - | - | - |

---

## 7. 주의사항 (중요)

이 포크에서 실제로 겪은 함정들. 새 실험 전에 반드시 확인한다.

### 7.1 C2f 는 upstream 원본 상태를 유지할 것

과거 구조적 프루닝을 위해 `C2f` 의 `cv1`(2c 채널)을 `cv0`+`cv1` 두 conv 로 분리한 패치가 있었다.
이 상태에서는 COCO 사전학습 체크포인트의 키가 매칭되지 않아 **가중치가 조용히 유실된다.**

| 상태 | yolov8n 가중치 전이 |
|------|--------------------|
| 프루닝 패치 적용 | `Transferred 315/403` ← 88개 유실 |
| **upstream 원복 (현재)** | `Transferred 355/355` ✓ |

C2f 는 backbone/neck 의 핵심 블록이라, 유실 상태로 학습하면 사전학습 효과를 크게 잃는다.
프루닝이 필요해지면 [docs_internal/C2F_PRUNING_PATCH.md](docs_internal/C2F_PRUNING_PATCH.md) 의 재적용 절차를 따르고,
`convert_weights.py` 로 체크포인트를 먼저 변환해야 한다.

### 7.2 VOC 에서 `Transferred 319/355` 는 정상

VOC 는 nc=20 이라 Detect head 의 cls 분기(`cv3`) 채널 수가 COCO(nc=80)와 달라진다.
해당 36개 항목은 새로 초기화되는 것이 정상이다. backbone·neck 은 전량 전이된다.
학습 로그에서 이 숫자를 항상 확인해 백본까지 유실되지 않았는지 점검한다.

### 7.3 `optimizer=auto` 는 100에폭에서 MuSGD 를 고른다

[`trainer.py`](ultralytics/engine/trainer.py) 의 분기:

```python
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

100에폭이면 iterations 가 5만~20만이라 **항상 MuSGD** 다(1~2에폭 테스트에서는 AdamW 가 선택되어 로그가 달라 보인다).
또한 `auto` 는 `lr0` / `momentum` 설정을 **무시**하고 MuSGD 기본값을 쓴다.
baseline 과 KD 실험이 모두 같은 조건을 타므로 비교에는 문제가 없다.

### 7.4 AutoBatch 는 과도하게 보수적

최악의 객체 수를 가정해 추정하므로 실측보다 훨씬 작은 값을 낸다(추천 12/4/2 vs 실측 32/16/8).
배치는 짧은 실측으로 정하는 편이 낫다. 다만 트레이너 밖에서 `autobatch()` 를 직접 호출하면
데이터셋 컨텍스트가 없어 프로파일링이 실패한다.

### 7.5 `project=` 에 `runs/detect` 를 넣지 말 것

[`cfg/__init__.py`](ultralytics/cfg/__init__.py) 의 `get_save_dir` 는 상대 경로 project 를 `RUNS_DIR/<task>/project` 로 합친다.

```python
project = args.project or ""
if not Path(project).is_absolute():
    project = RUNS_DIR / args.task / project      # runs/detect/<project>
save_dir = increment_path(Path(project) / name, ...)
```

따라서 `project="runs/detect/voc/baseline/yolov8n"` 은 `runs/detect/runs/detect/voc/baseline/yolov8n/train` 이 된다.
**dataset 부터 넘긴다** — `project="voc/baseline/yolov8n"`, `name="train"` → `runs/detect/voc/baseline/yolov8n/train`.

`RUNS_DIR` 는 ultralytics 전역 설정에 **절대 경로로 박혀 있다**(현재 `C:\Users\SSAFY\ultralytics\runs`).
따라서 결과 위치는 cwd 와 무관하다. 대신 리포지토리를 옮기면 이전 경로를 계속 가리키므로 다시 지정해야 한다.

```bash
yolo settings runs_dir=C:/Users/SSAFY/ultralytics/runs
```

### 7.6 val batch 는 train batch 의 2배 (하드코딩)

`trainer.py` 에서 `batch_size * 2` 로 고정되어 있고 별도 인자가 없다.
검증은 추론이라 메모리를 덜 쓰지만, 배치를 키울 때는 이 2배를 감안한다.
따로 조절하려면 `DetectionTrainer` 를 상속해 `get_dataloader` 에서 `mode == "val"` 일 때 배치를 바꾸고
`train(trainer=...)` 로 넘긴다.

한때 KD 스크립트가 이 방법으로 검증 배치를 8 로 캡했었다 — 검증 구간 VRAM 폭증 때문이었는데,
진범은 배치가 아니라 distiller 의 유령 hook 리크(§7.8)로 밝혀져 수정 후 캡을 제거했다.
오히려 캡은 역효과였다: 리크는 배치 크기가 아니라 **검증 배치 횟수**에 비례해서, 배치를 8 로
낮추자 forward 횟수가 4배로 늘어 누수도 4배가 됐다.

### 7.7 val 의 `rect` 기본값은 `True` 다 — `default.yaml` 을 믿으면 안 된다

`default.yaml` 에는 `rect: False` 로 적혀 있고 `train/args.yaml` 에도 `rect: false` 로 저장된다.
**그런데 val 에서 실제로 적용되는 값은 `True` 다.** `Model.val()` 이 method default 로 주입한다:

```python
# ultralytics/engine/model.py — Model.val()
custom = {"rect": True}  # method defaults
args = {**self.overrides, **custom, **kwargs, "mode": "val"}
```

학습 중 매 에폭 검증도 마찬가지로 `True` 다 — `DetectionTrainer.build_dataset` 이
`rect=mode == "val"` 로 넘긴다([detect/train.py](ultralytics/models/yolo/detect/train.py)).
`default.yaml` 의 `rect` 는 **학습용** 값이다.

재현성을 위해 `rect=False` 로 "기본값을 명시"하면 조용히 점수가 깎인다. 실측:

| 모델 | `rect=True` (실제 기본) | `rect=False` | 차이 |
|------|:---:|:---:|:---:|
| yolov8n | 0.6284 | 0.6243 | −0.0041 |
| yolov8s | 0.6750 | 0.6661 | −0.0089 |
| yolov8m | 0.7077 | 0.6975 | −0.0102 |

모델이 클수록 손해가 커진다. `rect=False` 는 모든 이미지를 640×640 정사각형으로 letterbox 해
패딩이 늘어나기 때문이다. 또한 이 값으로 재면 `train/results.csv` 의 에폭별 검증값과 비교가 안 된다.

**`VAL_ARGS` 에는 `rect=True` 를 박는다.** 그리고 "기본값과 같으니 명시해도 안전하다"는 가정을 하지 말고,
새 인자를 박을 때는 실측으로 확인한다.

> val 자체는 **결정적(deterministic)이다.** 같은 조건으로 두 번 돌리면 소수점 5자리까지 같은 값이 나온다.
> 따라서 `val/` 지표의 차이는 노이즈가 아니라 실제 차이다 — 재현이 안 되면 조건이 다른 것이다.

**그리고 `rect=True` 이면 `batch` 도 mAP 에 영향을 준다.** rect 는 이미지를 종횡비로 정렬한 뒤
**배치 단위로** letterbox 모양을 정한다:

```python
# ultralytics/data/base.py — set_rectangle()
bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
```

batch 가 바뀌면 한 배치에 묶이는 이미지 조합이 바뀌고, 그 배치의 종횡비 극단값으로 패딩이 정해지니
결과가 미세하게 달라진다. `rect=False` 라면 전부 640×640 이라 batch 는 mAP 에 영향이 없다.

따라서 **`batch` 는 VRAM 설정값이 아니라 평가 조건의 일부다.** `metrics.json` 에 남기는 이유가 여기 있다.
그리고 서로 비교할 런들은 val batch 를 같은 값으로 맞춰야 한다.

> 이 때문에 `train/results.csv` 의 마지막 에폭 값과 `val/` 값이 완전히 같지 않다.
> 학습 중 검증은 trainer 가 `batch*2` 로 돌리고(§7.6), `val/` 은 `VAL_ARGS` 의 batch 로 돌린다.
> best.pt 와 last.pt 차이에 이 rect 배치 차이가 더해진 것이다.

### 7.8 forward hook 이 걸린 모델을 deepcopy 하면 hook 도 복사된다 — EMA 메모리 리크

KD 첫 실행에서 에폭 말 검증 도중 메모리가 수십 GB 로 폭증해 시스템이 멎었다. 원인:

1. distiller 가 student 에 feature 캡처용 forward hook 을 걸고, **그 뒤에** `ModelEMA` 가 생성됐다.
2. `ModelEMA` 는 `deepcopy(model)` 로 만든다 — **`_forward_hooks` 까지 통째로 복사된다.**
3. 복사된 hook 의 storage 리스트도 deepcopy 로 분리된 별개 리스트가 된다. 학습 루프가 매 스텝
   비우는 건 원본 리스트라, **EMA 쪽 리스트는 아무도 비우지 않는다.**
4. 에폭 말 검증은 EMA 모델로 돈다 → 검증 배치마다 hook 이 발동해 feature 텐서가 무한 적재.

증상이 "검증 구간 VRAM 폭증"이라 검증 배치 크기를 의심하기 쉬운데, **리크는 배치 크기가 아니라
배치 횟수에 비례**한다. 배치를 낮추면 오히려 악화된다 (32→8 이면 forward 횟수 4배).

수정: hook 등록을 `_setup_train` 의 맨 끝(모든 EMA 생성·resume 이후)으로 옮겼다.
deepcopy 시점에 hook 이 존재하지 않으면 이 부류의 버그가 원천적으로 안 생긴다.
`save_model()` 의 hook 클리어는 안전망으로 남겨뒀다. 수정 전후 실측 (head1, batch 16):

| | peak reserved | 에폭 시간 |
|---|:---:|:---:|
| 수정 전 | 10.2G (스필) | 12.4분 |
| **수정 후** | **2.9G** | **7.8분** |

교훈: **hook 이 걸린 모델을 deepcopy 하는 경로(EMA, checkpoint 용 복사 등)가 있는지 항상 확인할 것.**
hook 은 모델 밖의 상태(storage)를 참조하는데, deepcopy 는 그 연결을 조용히 끊고 사본을 만든다.

### 7.9 Windows 에서는 `if __name__ == "__main__":` 필수

`workers > 0` 이면 DataLoader 가 spawn 으로 자식 프로세스를 만든다(Windows 는 fork 없음).
가드가 없으면 자식이 스크립트 전체를 재실행해 **프로세스가 무한 증식**한다.
학습 호출은 반드시 가드 안에 둔다.
