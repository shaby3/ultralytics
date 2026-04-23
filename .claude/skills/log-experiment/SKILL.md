---
name: log-experiment
description: "YOLOv8 KD 실험 완료 후 runs/detect/<experiment_name>/README.md를 처음 생성하고 runs/detect/EXPERIMENTS.md 비교 테이블과 베이스라인 대비 개선폭 테이블을 업데이트한다. 다음 같은 요청에는 반드시 이 skill을 사용: '실험 기록해줘', '학습 끝났어 정리해줘', '/log-experiment <path>', 'EXPERIMENTS.md 업데이트', '새 실험 결과 README 만들어줘', 'results.csv 기록해줘', 'KD 실험 문서화'. 사용자가 실험 디렉토리나 results.csv를 언급하며 결과 정리/기록/문서화를 원하면 명시적 키워드가 없어도 이 skill로 대응."
---

# log-experiment — YOLOv8 KD 실험 결과 자동 기록

YOLOv8n + VOC KD 실험의 반복적인 기록 작업을 자동화한다. `results.csv`에서 최종 metric을 추출하고, `args.yaml`에서 학습 파라미터를, `distill_cfg.yaml`에서 KD 설정을 추출해 실험별 README와 전체 비교 테이블(EXPERIMENTS.md)을 한 번에 업데이트한다.

## 인자

```
/log-experiment <experiment_dir>
```

- `<experiment_dir>`: 실험 결과가 있는 디렉토리 경로. repo root 기준 상대경로 또는 절대경로. 예: `runs/detect/neck_kd_v2`
- 인자가 누락되면 사용자에게 경로를 물어본다.

## 전제

- 현재 작업 디렉토리가 `C:\Users\SSAFY\ultralytics\` 또는 그 하위여야 한다 (EXPERIMENTS.md 경로 해석을 위해).
- 학습이 완료되어 `<experiment_dir>/results.csv`가 채워진 상태여야 한다.
- 기록 대상 실험은 처음 기록되는 것이어야 한다 (README.md가 없어야 한다). 이미 기록된 실험을 수정하는 것은 이 skill의 범위가 아니다.

## 포맷 기준 (Single Source of Truth)

이 프로젝트의 기존 실험 3개가 README와 EXPERIMENTS.md의 포맷 기준이다. 아래 템플릿만 보고 추론하지 말고, 헷갈릴 때는 항상 실제 파일을 `Read`해서 섹션 구조 / 테이블 컬럼 순서 / 표기 관례(소수점 자리, 구분자, 이모지 없음 등)를 맞추라 — 프로젝트의 기록 일관성이 분석 가치를 좌우하기 때문이다.

- `runs/detect/baseline_yolov8n/README.md` — baseline 템플릿 기준
- `runs/detect/neck_kd_yolov8n/README.md` — KD 템플릿 기준 (Neck-level, 3개 layer)
- `runs/detect/kd_yolov8n/README.md` — KD 템플릿 기준 (Head-level, 6개 layer)
- `runs/detect/EXPERIMENTS.md` — 전체 비교 테이블 기준

---

## Step 1: 입력 검증

1. `<experiment_dir>`가 실제로 존재하는 디렉토리인지 확인한다 (`Glob`로 `<experiment_dir>/*` 또는 `Bash`로 `test -d`).
2. `<experiment_dir>/results.csv`가 존재하는지 확인한다. 없으면:
   > "results.csv가 없습니다. 학습이 끝나지 않았거나 경로가 잘못되었습니다."
   로 중단한다.
3. `<experiment_dir>/README.md`가 **이미 존재하면** 중단한다:
   > "이 실험은 이미 기록되어 있습니다. 재작성이 필요하면 수동으로 처리해 주세요."

이 단계를 통과해야 이후 단계로 진행한다.

## Step 2: results.csv 파싱

`Read`로 `<experiment_dir>/results.csv`를 읽는다. Ultralytics 기본 헤더에서 다음 컬럼을 찾는다:

| 필요한 값 | 컬럼명 |
|-----------|--------|
| 완료 에폭 | `epoch` |
| mAP50 | `metrics/mAP50(B)` |
| mAP50-95 | `metrics/mAP50-95(B)` |
| Precision | `metrics/precision(B)` |
| Recall | `metrics/recall(B)` |
| KD 여부 판별 | `train/kd_loss` (있으면 KD 실험, 없으면 baseline) |

**마지막 비어있지 않은 행**에서 위 값을 추출한다. Metric 값은 **소수점 셋째 자리**로 반올림한다 (기존 README/EXPERIMENTS.md 표기 관례).

## Step 3: 학습 파라미터 추출

`Read`로 `<experiment_dir>/args.yaml`을 읽는다. Ultralytics는 학습 overrides를 이 파일에 자동 저장한다. 다음 필드를 추출:

- `model`, `data`, `epochs`, `batch`, `imgsz`, `optimizer`, `device`, `amp`, `pretrained`
- `distill_cfg` (있으면 KD 실험 확정, 없으면 baseline)

args.yaml이 없거나 필드가 누락되면 사용자에게 확인 질문을 한다.

## Step 4: KD 설정 추출 (KD 실험일 때만)

`args.distill_cfg` 경로(예: `ultralytics/cfg/distill_cfg.yaml`, `ultralytics/cfg/distill_neck_cfg.yaml`)를 `Read`로 읽어 다음을 추출:

- `teacher` — teacher 모델 weight 경로 (예: `yolov8s.pt`)
- 증류 대상 layer 목록 (layer index와 역할)
- `aligner` — aligner 타입 (예: `ConvAligner`)
- `loss` — distill loss 타입 (예: `MSE`)
- `weight` — distill loss weight

layer 설명은 기존 실험 README 스타일을 참고해 Claude가 초안을 만든다. 예:
- Head KD: `Detect head cv2(box) 0/1/2번 2nd conv, cv3(cls) 0/1/2번 2nd conv (총 6개)`
- Neck KD: `layer 15 (P3/8), layer 18 (P4/16), layer 21 (P5/32) — Neck output 3개`

## Step 5: 고유 정보 대화형 수집

자동 추출이 불가능한 두 가지 정보는 사용자에게 직접 묻는다. `AskUserQuestion`은 쓰지 않고 대화로 자연스럽게 요청한다:

1. **개요** — 이 실험의 의도를 한 문장으로. 예시:
   > "Neck-level Feature KD 실험. Teacher(yolov8s)의 Neck output(P3/P4/P5)을 Student(yolov8n)로 증류."
2. **EXPERIMENTS.md "KD 위치" 컬럼 요약** — 비교 테이블 용 짧은 요약. 예시:
   - KD 실험: `Neck output P3/P4/P5 (layer 15/18/21)`
   - baseline: `—`

KD 실험일 때는 Step 4에서 만든 layer 설명 초안도 함께 보여주고 사용자 확인/수정을 받는다.

## Step 6: README.md 생성

추출한 정보를 템플릿에 채워 `Write` 툴로 `<experiment_dir>/README.md`를 생성한다.

### Baseline 템플릿

```markdown
# <experiment_name>

## 개요
<Step 5에서 받은 개요>

## 세팅
| 항목 | 값 |
|------|-----|
| model | <args.model> |
| data | <args.data> |
| epochs | <args.epochs> |
| batch | <args.batch> |
| imgsz | <args.imgsz> |
| optimizer | <args.optimizer> |
| device | <args.device> |
| amp | <args.amp> |
| pretrained | <args.pretrained> |
| distill_cfg | 없음 |

## 결과
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| <mAP50> | <mAP50-95> | <Precision> | <Recall> |
```

### KD 템플릿

```markdown
# <experiment_name>

## 개요
<Step 5에서 받은 개요>

## 세팅
| 항목 | 값 |
|------|-----|
| model (student) | <args.model> |
| teacher | <distill_cfg.teacher> |
| data | <args.data> |
| epochs | <args.epochs> |
| batch | <args.batch> |
| imgsz | <args.imgsz> |
| optimizer | <args.optimizer> |
| device | <args.device> |
| amp | <args.amp> |
| pretrained | <args.pretrained> |
| distill_cfg | `<args.distill_cfg>` |

## KD 설정
| 항목 | 값 |
|------|-----|
| KD 레이어 | <Step 4/5에서 확정한 layer 요약> |
| aligner | <distill_cfg.aligner> |
| loss | <distill_cfg.loss> |
| weight | <distill_cfg.weight> |

## 결과 (<완료 에폭>/<args.epochs> 에폭 완료)
| mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| <mAP50> | <mAP50-95> | <Precision> | <Recall> |

> 다른 실험과의 비교는 [`../EXPERIMENTS.md`](../EXPERIMENTS.md) 참고.
```

`<experiment_name>`은 디렉토리 이름(basename)으로 채운다.

## Step 7: EXPERIMENTS.md 업데이트

`runs/detect/EXPERIMENTS.md`를 `Read`로 먼저 읽고 `Edit`로 두 군데를 수정한다.

### 7a. "결과 요약" 테이블에 행 추가

현재 "결과 요약" 표의 **마지막 행 바로 뒤**에 새 행을 추가:

```
| [<experiment_name>](./<experiment_name>/) | <KD 위치 요약 or —> | <args.batch> | <mAP50> | <mAP50-95> | <Precision> | <Recall> |
```

**중요 — 안전 삽입 패턴**: 기존 행 순서나 내용을 건드리지 않는다. 이유: 실험 기록의 이력 일관성이 깨지면 사람이 나중에 추적하기 어렵고, 다른 실험의 Δ 계산에 쓰인 기준값이 바뀌어 버릴 수 있다. 삽입은 `Edit`의 `old_string`에 현재 마지막 행 전체를 그대로 넣고, `new_string`에 `"기존 마지막 행\n새 행"`을 넣는 방식으로 한다. EXPERIMENTS.md를 `Write`로 재작성하는 건 금지 — "관찰" 섹션과 위의 업데이트 규칙, 그리고 사람이 손본 서식이 모두 날아간다.

### 7b. "베이스라인 대비 개선폭" 테이블에 행 추가 (KD 실험일 때만)

baseline 실험을 찾아 Δ 값을 계산한다:

1. "결과 요약" 테이블에서 "KD 위치" 컬럼이 `—`인 행을 찾는다 (예: `baseline_yolov8n`).
2. 해당 행의 mAP50, mAP50-95, Precision, Recall 값을 읽는다.
3. Δ = (현재 실험 값 − baseline 값) × 100, **소수점 1자리** 반올림, **부호 포함** (`+0.4`, `-0.1`).
4. "베이스라인 대비 개선폭" 표 마지막 행 뒤에 다음 형식으로 추가:

```
| <experiment_name> | <ΔmAP50> | <ΔmAP50-95> | <ΔPrecision> | <ΔRecall> |
```

baseline 실험(KD 위치 `—`)을 새로 기록하는 경우에는 7b 단계를 건너뛴다.

### 7c. "관찰" 섹션은 절대 건드리지 않는다

"관찰" 섹션은 여러 실험을 사람이 비교해 쓰는 서사이므로 자동 생성이 부적절하다. 사용자가 수동 유지한다.

## Step 8: 변경 요약 보고

마지막에 한 문단으로 다음을 보고한다:
- 생성한 README.md 경로
- EXPERIMENTS.md에 추가한 행 요약 (어느 표에 어떤 값으로)
- 사용자가 직접 업데이트해야 할 항목 안내: EXPERIMENTS.md의 "관찰" 섹션

예:
> `runs/detect/neck_kd_v2/README.md`를 생성했고 EXPERIMENTS.md 결과 요약과 베이스라인 대비 개선폭 테이블에 각각 한 행씩 추가했습니다. "관찰" 섹션은 여러 실험 비교가 필요해서 자동 업데이트하지 않았습니다 — 필요하면 직접 추가해 주세요.

---

## 실패 시 원칙

- 어느 단계에서든 필수 정보를 못 찾으면 **중단하고 사용자에게 묻는다**. 추측해서 잘못된 값을 기록하지 않는다.
- README.md는 한 번에 `Write`로 생성한다 (부분 쓰기 금지). EXPERIMENTS.md는 `Edit`로 정확한 위치에 삽입한다 (전체 재작성 금지 — 기존 내용 보호).
- EXPERIMENTS.md 편집이 실패하면 README.md는 이미 쓰인 상태일 수 있다. 이 경우 사용자에게 "README는 생성됐지만 EXPERIMENTS.md 업데이트가 실패했습니다. 수동으로 다음 행을 추가해 주세요: ..." 라고 알린다.
