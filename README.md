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

#### 실제 진행은 계획과 두 군데가 다르다

위 표는 **원래 계획**이다. 두 번 바꿨다.

**① Phase 1↔2 를 뒤집었다.** head`.1` 한 위치만 돌린 뒤 Phase 2 를 먼저 실행했고
(이유는 [§4](#phase-2-를-head1-에서-먼저-돌린다--계획된-순서에서-벗어난-결정이다)),
**COCO teacher 가 s-VOC 를 +1.53pt 로 이겨서** 남은 위치 스윕을 COCO teacher 위로 옮겼다.
그래서 **위치 비교는 s-VOC 가 아니라 s-COCO 위에서 성립한다.**

**② Phase 3 을 teacher 크기(m)에서 KD 기법으로 바꿨다.** 위치·teacher 두 축이 닫히고 나니
남은 질문 중 teacher 크기는 결과가 어느 쪽이든 해석이 정해져 있는 반면(크면 좋거나, capacity gap 이거나),
기법 축은 지금 남아 있는 방법론적 약점을 직접 건드린다 — MSE 는 위치마다 kd_loss 가 19~28배 벌어져
weight 를 손보정해야 했고([§4](#weight-는-위치별-kd_loss-실측으로-정규화했다)), 그 잔차가
Q2 결론에 교락으로 남아 있다. **teacher 크기는 최후순위 부록 ablation 으로 미룬다.**

| 순서 | 런 | 바뀌는 축 | 상태 |
|:---:|------|------|:---:|
| 1 | head`.1` × s-VOC | — | 완료 0.6342 |
| 2 | head`.1` × **s-COCO** | teacher 출처 | 완료 **0.6496** ← Q1 확정 |
| 3 | neck × s-COCO | 증류 위치 | 완료 0.6408 |
| 4 | head`.0` × s-COCO | 증류 위치 | 완료 0.6484 ← Q2 확정 |
| 5 | head`.1` × s-COCO × **PKD** | KD 기법 | 완료 **0.6533** ← 기법 선두 |
| 6 | head`.1` × s-COCO × **MGD** | KD 기법 | 완료 0.6501 (≈MSE) |
| 7 | head`.1` × s-COCO × **FGD** | KD 기법 | 대기 |
| — | head`.1` × **m**-COCO | teacher 크기 | 최후순위 |

1~4 번이 Q1·Q2 를 닫았고, 2번 런이 세 질문의 공통 기준점 역할을 한다 — Q1 의 대조쌍이자,
위치 스윕의 한 수준이자, 기법 스윕의 MSE 기준이다.

### 각 phase 가 허용하는 주장의 범위

**기법 축(Phase 3)은 head`.1`·s-COCO 위에서만 확인한다.** 네 수준 모두 위치·teacher·epochs·batch 가
같고 loss 또는 aligner 만 다르다. 다만 **MGD 와 FGD 는 한 축만 다른 비교가 아니다** —
MGD 는 정렬부·마스킹·생성 블록 셋이 함께 바뀌고 증류 경로 파라미터가 11.3배가 된다.
FGD 는 정렬부·attention 가중·GcBlock 이 함께 바뀌는 데다, 넥 전용으로 검증된 기법을 head 에 쓰는
**off-label** 적용이고, GT box 를 쓰는 유일한 수준이라 입력 정보량 자체가 다르다.
둘 다 방법의 정의라 피할 수 없고, MGD 가 이기면 λ=0 런으로 마스킹 순효과를 가른다
([§4](#phase-3--kd-기법-loss-와-aligner)). MGD 는 MSE 와 동률(+0.05pt)로 끝나 λ=0 런은 켜지지 않았다.

**Q3(teacher 크기)은 미뤘지만 비교 대상은 이미 확정됐다** — Phase 2 가 COCO 로 이겼으므로
**s-COCO(0.6496)** 와 한 축만 다른 비교가 되고, 그 런은 손에 있다.
teacher 크기 추이는 3점으로 읽는다: **teacher 없음(baseline n) → s → m.**
teacher 가 클수록 나빠지는 구간이 보이면 capacity gap 을 짚는 근거가 된다.

teacher 2×2 그리드(출처 × 크기)에서 **3칸이 채워지고 1칸(m-VOC)이 빈다.**
사전에 정해둔 규칙은 "COCO vs VOC 차이가 ~0.5pt 미만이면 빈 칸을 건너뛰고, 크면 1런 추가한다"였다.
실측 **+1.53pt** 는 큰 주효과이므로 **m-VOC 1런 추가가 규칙상 켜진다.**
"COCO-pretrained teacher 가 낫다"를 s 에서만 재고 일반화하면, m 에서 방향이 뒤집힐 때 논지가 무너지기 때문이다.
다만 m 런들은 전부 기법 축 뒤로 밀렸다.

**위치 순위는 s-COCO 에서만, 기법 순위는 head`.1` 에서만 확인한 것이다.** greedy 탐색의 한계다.
손에 있는 s-VOC head`.1` 런이 위치 쪽 부분적 방어는 해준다 —
teacher 를 바꿔도 head`.1` 이 여전히 상위라 교호작용이 약하다는 신호다.
필요하면 2위 설정을 최고 조합으로 1런 더 돌려 방어한다.

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

**위치 3수준.** teacher 는 셋 다 **s-COCO** 고정, aligner·loss 도 동일하고 `layers` 와 `weight` 만 다르다.

| distill config | KD 위치 | 지점 | student n → teacher s 채널 | weight |
|----------------|---------|:---:|----------------------------|:---:|
| `distill_neck_from_8s_coco.yaml` | neck 출력 (layer 15/18/21) | 3 | 64→128, 128→256, 256→512 | **20** |
| `distill_head0_from_8s_coco.yaml` | Detect head **1번째** conv | 6 | box 64→64 ×3, cls 64→128 ×3 | **10** |
| `distill_head1_from_8s_coco.yaml` | Detect head **2번째** conv | 6 | box 64→64 ×3, cls 64→128 ×3 | **1** (기준) |

각 config 에는 `teacher.model` 만 다른 `_from_8s_voc.yaml` 짝이 있다. **head`.1` 만 양쪽을 다 돌렸고**
(그게 Q1 이다), neck·head`.0` 의 `_voc` 판은 안 돌린다 — Q1 이 COCO 로 결론났기 때문이다.
지우지는 않는다. 교호작용을 방어해야 할 때 되살릴 대조군이다.

**COCO teacher(nc=80)와 VOC 학습본(nc=20)의 증류 지점 채널은 세 위치 전부 같다** — 실측 확인:

| | neck 15/18/21 | head`.0` box/cls | head`.1` box/cls |
|---|:---:|:---:|:---:|
| s-COCO (nc=80) | 128 / 256 / 512 | 64 / 128 | 64 / 128 |
| s-VOC (nc=20) | 128 / 256 / 512 | 64 / 128 | 64 / 128 |

neck 은 C2f 라 애초에 nc 와 무관하고, head 는 `c2 = max(16, ch0//4, reg_max*4) = 64`,
`c3 = max(ch0, min(nc,100))` 에서 s 는 `ch0=128` 이라 nc 가 80 이든 20 이든 128 이다.
따라서 **aligner 구조까지 동일**해서 teacher 출처만의 효과가 깨끗하게 분리된다.

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
teacher 출처도 마찬가지다 — head`.1` 의 1에폭 kd_loss 가 **VOC 6.14 → COCO 5.56** 으로 10% 움직였을 뿐이다.
**크기를 지배하는 건 teacher 가 아니라 위치다** — teacher 를 COCO 로 바꾸면서도 weight 를 재보정하지 않은 근거.
정규화의 목적이 위치 간 19배 자릿수 차이를 막는 것이므로 10% 변동은 무해하다.

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

각 스크립트는 baseline 스크립트와 같은 골격이다 — 학습 후 `best.pt` 재평가(`val/`)와
지표 기록(`results.csv` · `metrics.json`)까지 수행한다.
결과 경로 규칙은 [5. 결과 저장 구조](#5-결과-저장-구조) 참조.

#### Phase 2 를 head`.1` 에서 먼저 돌린다 — 계획된 순서에서 벗어난 결정이다

원래는 위치 3런을 끝내고 승자 위치에서 Phase 2 를 돌기로 했다. 순서를 바꾼 이유는
[6. 실험 결과](#두-회차의-비교--q1-의-예고편)의 두 회차 비교다 — 이전 회차 COCO-teacher 런이
절반의 에폭으로 더 높은 점수를 냈다. teacher 축이 예상보다 크다면 남은 위치 스윕(약 26시간)을
잘못된 teacher 위에서 쓰게 된다. 그래서 **teacher 를 먼저 확정하고 위치를 쓴다.**

- **COCO 가 크게 낫다** → 위치 스윕(neck·head`.0`)을 COCO teacher 로 진행, 순서를 재정렬한다.
- **비슷하다** → 원래 순서로 복귀. 이전 회차 차이는 epochs·batch 탓으로 정리된다.

**결과는 전자였다 — COCO +1.53pt** ([§6](#kd--위치-스윕-100에폭-batch-16)).
그래서 남은 두 위치는 `_from_8s_coco` 로 돌린다. 이미 끝난 `yolov8n_from_8s`(s-VOC) 런은
Q1 의 대조군으로 그대로 쓰이므로 낭비되는 런은 없다.

#### 실행

남은 위치 스윕 — baseline 과 마찬가지로 순차 실행한다 (동시 실행은 VRAM 부족):

```bash
.venv/Scripts/python.exe scripts/voc/kd_neck/train_yolov8n_from_8s_coco.py; .venv/Scripts/python.exe scripts/voc/kd_head0/train_yolov8n_from_8s_coco.py
```

이미 끝난 두 런 (재현용):

```bash
.venv/Scripts/python.exe scripts/voc/kd_head1/train_yolov8n_from_8s.py; .venv/Scripts/python.exe scripts/voc/kd_head1/train_yolov8n_from_8s_coco.py
```

> `scripts/voc/kd_{neck,head0}/train_yolov8n_from_8s.py`(s-VOC 판)도 남아 있지만 **돌리지 않는다.**
> Q1 이 COCO 로 결론나서, 이 둘을 돌리면 이미 진 teacher 위에서 위치를 비교하는 26시간이 된다.

Phase 3 은 승자 위치의 config·스크립트를 복사해 `teacher.model` 만 `yolov8m.pt` 로 바꾼다(`_from_8m`).

> 이전 회차(epochs 50)의 `run_phase6_kd*.py` · `run_neck_kd.py` 와 그 config 들은 삭제했다.
> 필요하면 git 히스토리에서 복구한다.

### Phase 3 — KD 기법 (loss 와 aligner)

위치·teacher 가 닫힌 뒤의 축이다. **head`.1` · s-COCO · 100에폭 · batch 16 고정** 위에서 4수준을 비교한다.

| 수준 | 무엇이 바뀌나 | config | weight | 상태 |
|---|---|---|:---:|:---:|
| **MSE** (기준) | — | `distill_head1_from_8s_coco.yaml` | 1.0 | 완료 0.6496 |
| **PKD** | loss 만 | `..._coco_pkd.yaml` | **8.0** | 완료 **0.6533** |
| **MGD** | aligner 만 (loss 는 mse 그대로) | `..._coco_mgd.yaml` | **1.4** | 완료 0.6501 |
| **FGD** | loss + aligner (아래 단서) | `..._coco_fgd.yaml` | **0.05** | 대기 |

**PKD** ([arXiv 2207.02039](https://arxiv.org/abs/2207.02039)) — feature 를 채널별로 평균 0·분산 1 로
표준화한 뒤 MSE. 수학적으로 `1 − r`(Pearson 상관)과 같다. `E[(x−y)²] = 1 − 2r + 1 = 2(1−r)` 이라 2로 나눈다.
**스케일에 무관해서 위치마다 kd_loss 가 19~28배 벌어지던 문제가 원천적으로 사라진다.**
대가는 teacher activation 의 크기 정보를 버리고 패턴만 증류한다는 것이다.

**MGD** ([arXiv 2205.01529](https://arxiv.org/abs/2205.01529)) — **loss 가 아니다.**
student feature 의 공간 위치를 λ 비율로 가린 뒤, 작은 conv block 이 살아남은 것만으로
teacher feature 를 *생성*하게 만든다. 마스킹과 생성 블록에 학습 파라미터가 있어 **aligner 자리**에 들어간다.
참조 구현(mmrazor)도 `MGDConnector`(파라미터 있음) + `MGDLoss`(평범한 MSE)로 쪼개져 있다.
구조는 그대로 따랐다 — 채널이 다를 때만 1×1 Conv 투영, 그 뒤 `Conv3×3 → ReLU → Conv3×3`.
head`.1` 6지점에서는 box(64→64)에 투영이 없고 cls(64→128)에만 생긴다.

참조 구현과 의도적으로 다른 점은 **스케일 관례 둘뿐이고 구조는 동일하다**: reduction 을 `sum/N` 대신
`mean` 으로 두고, `alpha_mgd` 대신 이 저장소의 `weight` 를 쓴다. 둘을 섞으면 정규화 체계가 둘이 되어
지금까지 위치별로 실측해 맞춘 weight 기준과 비교가 끊긴다.

**FGD** ([arXiv 2111.11837](https://arxiv.org/abs/2111.11837), CVPR 2022) — Focal + Global 두 부분의 합이다.
**Focal**: GT box 를 feature 에 투영해 전경/배경 마스크를 만들고, teacher 의 spatial·channel attention
(파라미터 없음 — 평균 절대값의 softmax)을 가중치로 전경과 배경을 **분리해서** MSE 를 건다
(`α·fg + β·bg`) + 두 attention 맵을 맞추는 L1(`γ·mask`). **Global**: GcBlock(`conv_mask` 1×1 →
softmax pooling → Conv-LN-ReLU-Conv)이 각자의 전역 문맥을 더한 뒤 MSE(`λ·rela`). GcBlock 은
student·teacher 각 1벌씩 **학습 파라미터**고, teacher 측도 학습된다 (입력이 detach 돼도 파라미터에는
grad 가 흐른다).

FGD 는 기존 두 슬롯 어디에도 안 맞아서 세 번째 구조가 필요했다. loss 에 학습 파라미터가 있는데
기존 loss 슬롯은 무상태 전제고, aligner 슬롯은 GT 와 teacher feature 를 못 받는다. 해법은
`DistillationWrapper` 에 유상태 loss 를 등록하는 것 — 옵티마이저가 wrapper 전체를 순회해 만들어지므로
(`_build_train_pipeline` → `build_optimizer(model=self.model)`) **aligner 가 학습되는 것과 같은 경로**로
GcBlock 이 학습된다. GT 는 KD loss 호출에 `batch=` 로 통과시킨다 (무상태 loss 는 무시).

구현은 레퍼런스([yzd-v/FGD](https://github.com/yzd-v/FGD) `fgd.py`)를 mmcv 의존만 제거하고 그대로
포팅했다 — 동일 가중치·입력에서 diff 0 을 확인했다. **내부 4항의 sum 기반 reduction 과 논문 기본값
(α=1e-3, β=5e-4, γ=1e-3, λ=5e-6)도 그대로 둔다.** 이는 "reduction 전부 mean" 관례의 의도적 예외다 —
이질적인 4항을 mean 으로 바꾸면 논문이 튜닝한 항간 비율이 근거 없이 뒤틀린다.
항간 비율은 논문값을 보존하고, **총량 스케일만 프로브 weight 로** 기존 기준에 맞춘다.
aligner 는 `IdentityAligner`(통과)다 — FGD 가 채널이 다를 때만 내부 1×1 투영을 만들기 때문에
슬롯에서 또 정렬하면 FGD 가 아니게 된다.

#### MGD·FGD 는 한 축만 다른 비교가 아니다

MSE·PKD 런 대비 **① 정렬부**(ConvBNSiLU → 1×1 Conv), **② 랜덤 마스킹**, **③ 생성 블록** 셋이 함께 바뀐다.
증류 경로 파라미터 실측(head`.1` 6지점 = box 64→64 ×3, cls 64→128 ×3):

| | 파라미터 | aligner 대비 | yolov8n 본체(3,157,200) 대비 |
|---|---:|---:|---:|
| `ConvBNSiLUAligner` (MSE·PKD) | 100,608 | 1× | 3.2% |
| `MGDAligner` (1×1 + 생성 블록) | 1,132,032 | **11.3×** | **35.9%** |
| FGD (align 1×1 ×3 + GcBlock 2벌 ×6) | 151,884 | 1.51× | 4.8% |

MGD 의 배율은 커널(3×3 vs 1×1)에서 9배, 폭에서 1.3배가 곱해진 값이다. 추론 때 버려지지만 학습 중
증류 경로에 그만한 용량이 얹힌다. **MGD 라는 방법의 정의라 피할 수 없다** — 결과에 이 단서를 달고,
MGD 가 이기면 `aligner_args: {lambda_mgd: 0.0}` 런(생성 블록은 두고 마스킹만 끔)을 추가해
**λ=0.65 vs λ=0 차이 = 마스킹 순효과**로 가른다.
→ **결과: MGD 0.6501 ≈ MSE 0.6496 (+0.05pt, 노이즈 안).** 이기지 못했으므로 λ=0 런은 켜지 않는다.

FGD 는 용량 문제는 작지만(1.51×) 다른 세 가지가 겹친다. **① off-label** — FGD 는 넥(FPN) feature
전용으로 설계·검증됐고 head 중간 특징에 대한 논문 근거가 없다. 기법 축을 유지하기 위한 의도적
선택이고, 지면 기법 탓인지 위치 탓인지 가를 수 없다. **② 다축 변화** — 정렬부(ConvBNSiLU → FGD 내부
1×1)·attention 가중·GcBlock 이 함께 바뀐다. **③ 입력 정보량** — GT box 를 쓰는 유일한 수준이라
feature 만 보는 나머지 셋과 증류에 들어가는 정보 자체가 다르다 (이기면 이것도 원인 후보다).

#### weight 는 여기서도 프로브로 맞춘다

기법마다 kd_loss 자릿수가 다르므로 위치 스윕과 같은 규칙을 쓴다 — **MSE 런의 초기 KD 비중(54.3%)에 맞춘다.**

```
w = (kd_mse / task_mse) × task_method / kd_method       (w_mse = 1)
```

기준값은 과거 기록이 아니라 프로브의 mse 런에서 새로 뽑는다. 같은 세션·같은 조건이라 비교가 깨끗하다.
task loss 가 mse/pkd/mgd 에서 4.554/4.554/4.556 으로 사실상 같아, 식은 실질적으로 **`w = kd_mse / kd_method`** 다
(fgd 만 예외 — 아래 두 번 잰 이야기 참조).

FGD 통합 리팩터(KD loss 호출 시그니처 변경) 후 mse 를 재프로브했더니 kd_loss 가 이전 측정과
**소수점 15자리까지 동일**하게 재현됐다 — 학습이 결정적이라 가능한, 기존 경로 무변경의 가장 강한 증거다.

```bash
.venv/Scripts/python.exe scripts/voc/probe_kd_scale.py
```

결과는 `scripts/voc/probe_kd_scale.json` 에 런마다 즉시 저장되고 이미 잰 항목은 건너뛴다 —
중간에 죽어도 재실행하면 이어붙는다. **실측 (batch 16, 각 ~10분):**

| 기법 | ep1 kd_loss | KD 비중 (w=1) | 산출 w | **채택 w** | 보정 후 KD 비중 | peak VRAM | 분/에폭 |
|---|---:|---:|---:|:---:|---:|---:|---:|
| mse | 5.2608 | 53.6% | 1.000 | **1.0** | 53.6% | 2.68 GB | 9.9 |
| pkd | 0.6476 | 12.4% | 8.124 | **8.0** | 53.2% | 2.84 GB | 10.2 |
| mgd | 3.7226 | 45.0% | 1.414 | **1.4** | 53.4% | 2.59 GB | 10.2 |
| fgd | 117.53 | 96.0% | 0.0477 | **0.05** | 54.8% | 2.90 GB | 10.4 |

네 기법의 초기 KD 비중이 **53.2~54.8%, 1.6pt 안에 모였다.** PKD 는 `1 − r` 이라 1 부근에 갇혀 MSE 의 1/8 이고,
정규화 없이 돌렸으면 12.4% 로 사실상 KD 를 절반쯤 끈 셈이 됐을 것이다.
MGD 가 같은 MSE 인데도 낮은 건 생성 블록이 teacher feature 를 직접 맞추도록 학습되기 때문이다.
FGD 는 반대 극단이다 — sum reduction 이라 task 의 26배이고, 정규화 없이는 KD 비중 96% 로
학습이 사실상 KD 만 남는다. **정규화 규칙이 1/8 배(pkd)부터 26배(fgd)까지, 자릿수로 3칸을 한 기준에 묶었다.**

> **FGD 만 프로브를 두 번 돌렸다.** w=1.0 첫 프로브(JSON 의 `fgd_w1p0`)에서 KD 비중이 95% 가 되어
> task_loss 까지 오염됐다(6.15 vs 정상 4.55) — 공식의 task 항이 오염되면 w 가 35% 과대추정된다.
> 보정 추정치 0.046 으로 재프로브해 정상 조건에서 확정했다. kd_loss 자체는 두 측정에서 115.1/117.5 로
> 2% 차이라 weight 의존성이 거의 없었고, 오염은 task 쪽에만 있었다.
> 다른 기법은 w=1.0 프로브에서도 KD 비중이 12~54% 라 이 문제가 없었다.

**MGD 의 비용은 예상보다 작다** — 에폭 시간이 MSE 대비 +3%, VRAM 은 넷 다 2.6~2.9GB 로
6.1GB 대비 여유가 있다. batch 16 을 그대로 간다. FGD 는 GT 마스크 계산 때문에 처음엔 +97%(17.9분)였는데,
per-box 파이썬 루프를 브로드캐스트 max 로 벡터화해(수식 동일 — 레퍼런스와 diff 0 유지) +15%(10.4분)로
내렸다. 100에폭 기준 약 12시간 절약이다.

> **프로브는 AdamW 로 돌았다** — `optimizer=auto` 가 `iterations > 10000` 에서만 MuSGD 를 고르는데
> 1에폭은 1,035 iteration 이다 ([§7.3](#73-optimizerauto-는-100에폭에서-musgd-를-고른다)).
> 그래서 mse 의 ep1 kd_loss 가 본 런(MuSGD) 기록 5.5596 이 아니라 5.2608 로 나왔다.
> **정규화에는 영향이 없다** — 프로브가 모두 같은 AdamW 아래라 비율이 상쇄되고,
> 기준을 본 런 값으로 바꿔 계산해도 weight 는 8.12→8.37 / 1.41→1.46 으로 **3.0% 움직일 뿐**이다.
> 이 정규화가 이미 감수하는 반올림 폭(18.8→20 = 6%)보다 작다.

#### 실행

```bash
.venv/Scripts/python.exe scripts/voc/kd_head1/train_yolov8n_from_8s_coco_fgd.py
```

(PKD·MGD 런은 완료 — 결과는 [§6](#6-실험-결과).)

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
      yolov8n_from_8s/        #   teacher s-VOC          — 완료 0.6342
      yolov8n_from_8s_coco/   #   teacher 출처만 교체     — 완료 0.6496  ← Q1 대조쌍
        train/  val/
      yolov8n_from_8m_coco/   #   teacher 크기만 교체     — 대기 (Phase 3)
    kd_head0/
      yolov8n_from_8s_coco/   #   대기 — 위치 스윕
    kd_neck/
      yolov8n_from_8s_coco/   #   대기 — 위치 스윕
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

#### 기법 축은 variant 뒤에 붙인다

Phase 3 에서 축이 하나 늘었다. `yolov8n_from_8s_coco_pkd` / `_mgd` / `_fgd` 처럼 **teacher 뒤에 기법을 붙인다.**

접미사가 없으면 MSE 다 — teacher 규칙과 같은 논리로, 기준 수준을 접미사 없는 쪽에 둔다.
표면상 variant 가 teacher 와 기법 두 축을 담는 것처럼 보이지만, **기법 스윕 안에서 teacher 는 s-COCO 로
고정이라 실제로 갈리는 축은 기법 하나**다. method 를 `kd_head1_pkd` 로 쪼개지 않는 이유는
method = 증류 위치 규칙이 깨지기 때문이다.

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

### KD — 위치 스윕 (100에폭, batch 16)

`val/` 의 `best.pt` 재평가 기준. Δ 는 baseline n(100에폭, 0.6284) 대비.

| 위치 | teacher | weight | mAP50 | mAP50-95 | Precision | Recall | ΔmAP50-95 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| head`.1` | s-VOC | 1 | 0.8362 | 0.6342 | 0.8241 | 0.7568 | +0.58pt |
| head`.1` | **s-COCO** | 1 | 0.8499 | **0.6496** | 0.8287 | 0.7732 | **+2.12pt** |
| head`.0` | s-COCO | 10 | 0.8471 | 0.6484 | 0.8208 | 0.7716 | +2.00pt |
| neck | s-COCO | 20 | 0.8410 | 0.6408 | 0.8217 | 0.7659 | +1.24pt |

네 런 모두 `epochs=100, batch=16, imgsz=640, patience=30, amp=True`, aligner `ConvBNSiLUAligner`,
loss `mse` 로 동일하고 `layers` 와 `weight` 만 다르다. early stopping 은 어디서도 걸리지 않았다.

### Q2 확정 — head`.0` ≈ head`.1` > neck

**head`.1` 과 head`.0` 은 0.11pt 차이라 구분되지 않는다.** [§1 의 기준](#결과-해석-시-주의)(0.3pt)의 1/3 이다.
궤적을 보면 더 분명하다 — 두 런이 계속 자리를 바꾼다:

| epoch | 10 | 30 | 50 | 70 | 90 | 100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| neck | 0.4657 | 0.5918 | 0.6263 | 0.6352 | 0.6385 | 0.6408 |
| head`.0` | 0.4998 | 0.6050 | 0.6374 | 0.6437 | 0.6467 | 0.6480 |
| head`.1` | 0.5093 | 0.6115 | 0.6354 | 0.6440 | 0.6467 | 0.6489 |
| head`.1` − head`.0` | +0.95pt | +0.65pt | **−0.20pt** | +0.03pt | −0.00pt | +0.09pt |

**neck 과의 0.76pt 차이는 실재한다.** ep20 이후 전 구간에서 head 두 위치가 neck 위에 있고
격차가 부호를 바꾼 적이 없다.

**weight 정규화는 제 역할을 했다.** ep1 실측 기준 실효 KD 기여:

| 위치 | ep1 kd_loss | ×weight | task loss | KD 비중 | (weight=1 이었다면) |
|---|:---:|:---:|:---:|:---:|:---:|
| neck | 0.200 | 4.01 | 4.78 | 45.6% | 4.0% |
| head`.0` | 0.437 | 4.37 | 4.71 | 48.1% | 8.5% |
| head`.1` | 5.560 | 5.56 | 4.67 | 54.3% | 54.3% |

정규화가 없었으면 4.0% / 8.5% / 54.3% 였을 것이 **45.6~54.3% 안으로 모였다.** 다만 완전히 같지는 않고
head`.1` 이 neck 보다 1.19배 강하다. 그래서 정확한 표현은 "neck 이 나쁘다"가 아니라
**"neck 은 같은 정도의 KD 신호로 head 만큼 얻어내지 못한다"** 다.
반대로 neck vs head`.0` 은 45.6% vs 48.1% 로 거의 같아 **이 쌍이 위치 효과를 가장 깨끗하게 보여준다**(0.76pt).

> **네 런 모두 100에폭에서 아직 오르는 중이다.** best epoch 가 100/100/99/100 이고 patience 30 이
> 한 번도 안 걸렸다(baseline 은 ep99 정점 후 평탄). 위 절대 수치는 하한이다.
> 위치 순위가 0.3pt 안에 몰린 게 "구분 안 됨"이 아니라 "아직 안 갈림"일 수 있다.

**한계 — 위치 순위는 s-COCO 에서만 확인한 것이다.** 다른 teacher 에서 같은 순위인지는 검증하지 않았다.
손에 있는 s-VOC head`.1` 런이 부분적 방어는 해준다(teacher 를 바꿔도 head`.1` 은 여전히 상위다).

### Q1 확정 — COCO-pretrained teacher 가 VOC 학습본을 +1.53pt 로 이긴다

위 두 head`.1` 런은 **`teacher.model` 한 줄만 다른 통제쌍**이다. 위치·aligner·loss·weight·epochs·batch·seed 가 전부 같고,
증류 지점 채널까지 동일해 aligner 구조도 같다([§4](#kd-학습)). **teacher 출처 하나만 분리된 비교다.**

세 가지가 이 차이를 노이즈에서 떼어놓는다.

**1. 학습 전 구간에서 앞선다.** 최종 시점만의 우연이 아니다 (`train/results.csv`, mAP50-95):

| epoch | 10 | 30 | 50 | 70 | 90 | 100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline n | 0.4104 | 0.5754 | 0.6169 | 0.6253 | 0.6283 | 0.6288 |
| head`.1` s-VOC | 0.4797 | 0.5945 | 0.6227 | 0.6313 | 0.6333 | 0.6342 |
| head`.1` **s-COCO** | **0.5093** | **0.6115** | **0.6354** | **0.6440** | **0.6467** | **0.6489** |
| COCO − VOC | +2.96pt | +1.70pt | +1.27pt | +1.27pt | +1.34pt | +1.47pt |

초반 격차가 크고(ep10 +2.96pt) 이후 +1.3pt 부근에서 안정된다.
수렴 후에도 격차가 유지된다는 건 "COCO teacher 가 빨리 배우게 한다"가 아니라 **도달 지점이 다르다**는 뜻이다.

**2. 20개 클래스 중 19개가 개선됐다.** 유일한 하락은 cat −0.14pt 로 측정 노이즈 수준이다.
개선 폭 상위는 tvmonitor +3.24, boat +2.77, bottle +2.74, pottedplant +2.65 — **작고 어려운 객체 위주**다.
COCO 80클래스 사전학습이 준 표현 다양성이 VOC 소수 클래스로 전이됐다는 해석과 맞는다.

**3. 격차가 seed 노이즈 규모를 넘는다.** +1.53pt 는 [§1 에서 경계하기로 한 수준](#결과-해석-시-주의)(~0.3pt)의 5배다.

kd_loss 도 COCO 쪽이 낮게 수렴했다 (0.976 vs 1.088) — student 가 COCO teacher feature 를 더 잘 따라갔다.

**해석.** VOC 로 fine-tune 한 teacher 는 그 데이터셋에서 mAP 가 더 높지만(s-VOC 0.6750),
**증류 대상으로서는 COCO-pretrained 쪽이 낫다.** 20클래스에 특화되며 좁아진 feature 보다
80클래스에서 만들어진 일반적 feature 가 student 에게 더 풍부한 신호를 준다는 것이다.
"teacher 의 task 성능"과 "teacher 의 증류 가치"가 같은 방향이 아니라는, 이 실험의 가장 뾰족한 결과다.

**한계 — 한 위치(head`.1`)에서만 확인했다.** 남은 위치 스윕을 COCO 로 옮긴 결정은
teacher 출처 × 위치 교호작용이 없다는 미검증 가정 위에 있다. 논문에는 범위를 명시해야 한다.

> **두 KD 런 모두 100에폭에서 아직 오르는 중이다.** best epoch 가 정확히 100 이고 patience 30 이 한 번도 안 걸렸다
> (baseline 은 ep99 에서 정점 후 평탄). **KD 런은 미포화 상태**라 위 절대 수치는 하한이고,
> 에폭을 늘리면 격차가 더 벌어질 수 있다. 세 런의 조건이 같으므로 비교 자체는 유효하다.

### KD — 기법 스윕 (head`.1` × s-COCO 고정, 100에폭, batch 16)

`val/` 의 `best.pt` 재평가 기준. Δ 는 MSE 런(기법 축의 기준, 0.6496) 대비.

| 기법 | weight | mAP50 | mAP50-95 | Precision | Recall | ΔmAP50-95 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MSE (기준) | 1.0 | 0.8499 | 0.6496 | 0.8287 | 0.7732 | — |
| **PKD** | 8.0 | 0.8565 | **0.6533** | 0.8303 | 0.7844 | **+0.37pt** |
| MGD | 1.4 | 0.8519 | 0.6501 | 0.8224 | 0.7837 | +0.05pt |
| FGD | 0.05 | — | — | — | — | 대기 |

**PKD 의 +0.37pt 는 실재하지만 작다.** 100에폭 전부에서 MSE 를 앞섰고(100/100) 격차가 부호를 바꾼 적이
없다 — 이 일관성이 최종값 차이보다 강한 근거다. 다만 크기는 [§1 노이즈 기준](#결과-해석-시-주의)(0.3pt)을
살짝 넘는 수준이고, 클래스별로는 20개 중 13개 개선이라 전면적이지 않다 (개선 상위: bus +1.78,
bottle +1.71, dog +1.62 / 하락: sheep −0.71, aeroplane −0.69). 순위를 단단히 하려면 seed 반복이 필요하다.

**MGD 는 MSE 와 동률이다.** +0.05pt 는 노이즈 기준의 1/6 이고, 클래스별 개선 11/20 은 동전 던지기
수준이다. 에폭별 우세 85/100 로 후반에 근소하게 앞서긴 하지만 격차가 커지지 않는다.
**11.3배의 증류 경로 용량이 이 위치에서는 아무것도 사주지 않았다** — λ=0 통제 런은
"MGD 가 이기면"이 조건이었으므로 켜지 않는다.

kd_loss 궤적은 두 기법의 성격 차이를 보여준다 — MSE·MGD 는 0.98 까지 계속 내려가는 반면
PKD 는 ep20 부터 0.17~0.19 (r ≈ 0.82) 에서 평탄하다. **student 용량으로 도달 가능한 상관 상한에
일찍 닿고, 이후는 패턴 유지 상태로 학습이 진행된다**는 뜻이다. 그런데도 최종 성능은 PKD 가 가장 높다 —
teacher activation 의 크기까지 맞추는 것(MSE)보다 채널별 패턴만 맞추는 것(PKD)이
이 위치에서는 더 나은 신호라는 증거다.

세 런 모두 `epochs=100, batch=16, imgsz=640, patience=30, amp=True` 동일, early stopping 없음
(best epoch 100/99/100). 기법 축의 해석 한계(MGD·FGD 의 다축 변화, FGD 의 off-label)는
[§4](#mgdfgd-는-한-축만-다른-비교가-아니다) 참조.

### KD — 이전 회차 (50에폭, batch 32, teacher **COCO-pretrained** `yolov8s.pt`)

git 히스토리에서 복원했다(`8e45746f~1`). 당시엔 weight 정규화도 aligner 고정도 없었다.
Δ 는 같은 회차 baseline n(50에폭 **batch 16**, 0.6121) 대비 — batch 가 달라 교락이 있다.

| 위치 | aligner | weight | mAP50 | mAP50-95 | 최종 `kd_loss` | ΔmAP50-95 |
|------|---------|:---:|:---:|:---:|:---:|:---:|
| head`.1` | ConvBNSiLU | 1 | 0.8430 | **0.6428** | 0.974 | +3.06pt |
| head`.1` | ConvAligner | 1 | 0.8428 | 0.6398 | 0.966 | +2.76pt |
| neck | ConvAligner | 1 | 0.8214 | 0.6193 | **0.038** | +0.71pt |

**이전 회차의 "neck 이 head 보다 나쁘다"는 결과는 위치 효과가 아니다.** neck 의 최종 `kd_loss` 가
0.038 로 head1(0.974)의 1/25 다 — `weight=1.0` 이 neck 에서는 사실상 KD 를 끈 것과 같았다.
현재 회차가 weight 를 정규화한 이유이고([§4](#weight-는-위치별-kd_loss-실측으로-정규화했다)),
그래서 Q2 는 아직 답이 나오지 않은 열린 질문이다.

### 두 회차의 비교 — Q1 의 예고편

| | 이전 회차 head`.1` | 현재 회차 head`.1` |
|---|:---:|:---:|
| teacher | **COCO-pretrained** | **VOC 학습본** |
| epochs / batch | 50 / 32 | 100 / 16 |
| aligner / weight | ConvBNSiLU / 1 | ConvBNSiLU / 1 |
| **mAP50-95** | **0.6428** | **0.6342** |

**절반의 에폭으로 COCO teacher 쪽이 0.86pt 높다.** "VOC 로 fine-tune 한 teacher 가 당연히 낫다"는
가정과 반대다. 다만 teacher 말고도 epochs·batch 가 함께 달라 이것만으로는 결론지을 수 없었다.

**이 예고편은 맞았다.** teacher 만 바꾼 통제 런(위 [Q1 확정](#q1-확정--coco-pretrained-teacher-가-voc-학습본을-153pt-로-이긴다))이
같은 방향으로, 더 큰 폭(+1.53pt)으로 재현했다. 즉 위 표의 0.86pt 는 teacher 효과가
epochs·batch 차이에 **일부 상쇄된** 값이었다.

그 상쇄가 무엇이었는지도 이제 읽힌다. baseline 은 50→100에폭에서 0.6121→0.6284(**+1.63pt**) 올랐는데
이전 회차 KD 는 0.6428, 현재 회차 COCO-KD 는 100에폭에 0.6496 이다 — 같은 teacher 기준 **+0.68pt**.
**긴 스케줄이 KD 이득을 희석하는 건 맞다**(baseline 이 +1.63pt 올라올 때 KD 는 +0.68pt 만 올랐다).

이 분해는 **잠정적이다.** batch(32→16)가 함께 다르고, 이전 회차 수치는 `val/` 이 없어 `train/` 마지막 에폭 값이다
(현재 회차 기준 둘의 차이는 0.0005 미만이라 결론을 바꿀 크기는 아니다).
Q1 결론에는 영향이 없다 — Q1 의 두 런은 둘 다 100에폭·batch 16·`val/` 재평가값이다.

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

### 7.10 distill config 의 `loss` 오타가 조용히 MSE 로 떨어졌다 (수정됨)

`_setup_kd_loss` 가 이랬다:

```python
loss_map = {"mse": nn.MSELoss(reduction="mean")}
self.kd_loss_fn = KDFeatureLoss(loss_fn=loss_map.get(loss_name))   # 미등록 이름 -> None
```

`loss_map.get()` 이 `None` 을 주고, `KDFeatureLoss.__init__` 의 `loss_fn or nn.MSELoss(...)` 가
그걸 **MSE 로 되돌린다.** 그래서 `loss: pkd` 라고 써도 오류 없이 학습이 돌고,
13시간 뒤 결과를 PKD 로 착각하게 된다. 같은 파일의 `_setup_aligner` 는 이름이 틀리면
`ValueError` 를 던지는데 loss 쪽만 안 막혀 있었다.

**침묵하는 기본값은 실험 코드에서 특히 나쁘다** — 오타가 실패가 아니라 *다른 실험*으로 나타난다.
등록되지 않은 이름은 aligner 와 같은 방식으로 거부하도록 고쳤다.

```python
if loss_name not in loss_map:
    raise ValueError(f"Unknown KD loss '{loss_name}'. Available losses: {sorted(loss_map)}")
```

`KDFeatureLoss` 의 `or` 기본값 자체는 그대로 뒀다 — `loss_fn=None` 의 문서화된 동작이고,
상류에서 막으면 충분하다.

### 7.11 loss 슬롯의 학습 파라미터는 wrapper 등록 없이는 조용히 얼어붙는다

옵티마이저는 `_build_train_pipeline` → `build_optimizer(model=self.model)` 로 **`self.model` 을
순회해서만** 만들어진다. aligner 가 학습되는 것도 `DistillationWrapper` 에 실려 있기 때문이지,
aligner 라서가 아니다.

FGD 처럼 **loss 모듈에 학습 파라미터가 있는 기법**(GcBlock 2벌 + 내부 align conv)을
`self.kd_loss_fn = FGDFeatureLoss(...)` 로만 두면 — 아무 오류 없이 학습이 돌고, kd_loss 도 찍히고,
grad 도 계산되지만 **옵티마이저가 그 파라미터를 모르므로 step 이 적용되지 않는다.**
GcBlock 이 초기값에 얼어붙은 채 13시간을 돌고, 결과를 "FGD 를 적용한 것"으로 착각하게 된다.
7.10 의 MSE 폴백과 같은 부류다 — 오타/누락이 실패가 아니라 *다른 실험*으로 나타난다.

그래서 `_setup_kd_loss` 를 wrap **전에** 호출하도록 순서를 바꾸고, `nn.Module` 인 loss 는
wrapper 에 등록한다:

```python
kd_loss_module = self.kd_loss_fn if isinstance(self.kd_loss_fn, nn.Module) else None
self.model = DistillationWrapper(self.model, self.aligner_module, kd_loss_module)
```

검증도 이 지점을 겨눴다 — FGD 파라미터 102개 전부가 `wrapper.parameters()` 에 포함되는 것을
확인하는 테스트가 이 통합에서 가장 중요한 항목이다. EMA·checkpoint(`kd_loss_last.pt`)도
같은 경로를 탄다. 무상태 loss(mse/pkd)는 `nn.Module` 이 아니어서(`KDFeatureLoss`) 기존과 동일하다.
