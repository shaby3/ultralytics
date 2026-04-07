# Knowledge Distillation Framework - TDD Implementation Plan

## Context
Ultralytics YOLO 코드베이스에 feature-level Knowledge Distillation(KD) 프레임워크를 추가한다.
Teacher/Student 모델의 distillation point를 YAML로 자유롭게 지정하고, aligner를 통해 feature 차원을 매핑한 후 MSE loss로 학습한다.
KD 기능은 평상시 비활성화 상태이며, distill_cfg가 지정될 때만 동작한다.

## 기본 설정
- **Student**: yolov8n.yaml
- **Teacher**: yolov8s.pt
- **Data**: coco8.yaml
- **Distillation points**: Detect head(index 22) 내부 cv2(box)/cv3(cls) 브랜치의 2번째 Conv 출력

### YOLOv8 Detect Head 구조 (index 22)
```
model.model[22] = Detect(from=[15, 18, 21])
  ├── cv2 (box branch): 3 scales × Sequential(
  │     [0] Conv(x, c2, 3),       # 1번째 conv
  │     [1] Conv(c2, c2, 3),      # 2번째 conv ← distillation point
  │     [2] Conv2d(c2, 4*reg_max, 1)  # 출력 projection
  │   )
  └── cv3 (cls branch, legacy=True): 3 scales × Sequential(
        [0] Conv(x, c3, 3),       # 1번째 conv
        [1] Conv(c3, c3, 3),      # 2번째 conv ← distillation point
        [2] Conv2d(c3, nc, 1)     # 출력 projection
      )
```

### Distillation Points (6개)
| Branch | Scale | Module Path | 출력 채널 |
|--------|-------|-------------|-----------|
| box | P3/8 | `model.22.cv2.0.1` | c2 |
| box | P4/16 | `model.22.cv2.1.1` | c2 |
| box | P5/32 | `model.22.cv2.2.1` | c2 |
| cls | P3/8 | `model.22.cv3.0.1` | c3 |
| cls | P4/16 | `model.22.cv3.1.1` | c3 |
| cls | P5/32 | `model.22.cv3.2.1` | c3 |

---

## 수정/생성 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `ultralytics/nn/modules/aligner.py` | **신규** | ConvAligner, MultiScaleAligner 모듈 |
| `ultralytics/nn/modules/__init__.py` | 수정 | aligner 모듈 import/export 추가 |
| `ultralytics/utils/loss.py` | 수정 | KDFeatureLoss 클래스 추가 |
| `ultralytics/cfg/default.yaml` | 수정 | distill_cfg 키 추가 (기본 비활성화) |
| `ultralytics/engine/distiller.py` | **신규** | DistillationWrapper, create_distiller 팩토리 |
| `ultralytics/engine/__init__.py` | 수정 | distiller export 추가 |

---

## Distillation YAML 설정 파일 구조

별도의 `distill_cfg.yaml` 파일로 KD 관련 설정을 모두 관리:

```yaml
# distill_cfg.yaml (YOLOv8 head 내부 cv2/cv3 2번째 conv 증류)
teacher:
  model: yolov8s.pt
  layers:
    - "model.22.cv2.0.1"   # box branch scale0 2nd conv
    - "model.22.cv2.1.1"   # box branch scale1 2nd conv
    - "model.22.cv2.2.1"   # box branch scale2 2nd conv
    - "model.22.cv3.0.1"   # cls branch scale0 2nd conv
    - "model.22.cv3.1.1"   # cls branch scale1 2nd conv
    - "model.22.cv3.2.1"   # cls branch scale2 2nd conv

student:
  layers:
    - "model.22.cv2.0.1"
    - "model.22.cv2.1.1"
    - "model.22.cv2.2.1"
    - "model.22.cv3.0.1"
    - "model.22.cv3.1.1"
    - "model.22.cv3.2.1"

aligner: ConvAligner
loss: mse
weight: 1.0
```

### Layer 지정 방식 (혼합 지원)
- **정수**: `model.model[idx]`의 출력을 캡처 (top-level Sequential 레이어)
- **문자열**: `model.get_submodule(path)`로 중첩 모듈의 출력 캡처 (예: `"model.22.cv2.0.1"`)
- teacher/student의 `layers` 수는 반드시 동일해야 함 (1:1 매핑)
- layers 미지정 시 기본값으로 `model.model[-1].f` (head 입력 인덱스) 사용

---

## 파일별 구현 상세

### 1. `ultralytics/nn/modules/aligner.py` (신규)

```python
import torch.nn as nn


class ConvAligner(nn.Module):
    """1x1 Conv -> ReLU -> 1x1 Conv. Student feature를 teacher feature 차원에 매핑."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        return self.align(x)


class MultiScaleAligner(nn.Module):
    """여러 distillation point에 대해 각각 ConvAligner를 생성/관리."""

    def __init__(self, student_channels, teacher_channels, aligner_cls=ConvAligner):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)
        self.aligners = nn.ModuleList(
            [aligner_cls(sc, tc) for sc, tc in zip(student_channels, teacher_channels)]
        )

    def forward(self, features):
        """features: list[Tensor] - distillation point별 student feature maps."""
        return [aligner(feat) for aligner, feat in zip(self.aligners, features)]
```

### 2. `ultralytics/nn/modules/__init__.py` (수정)
- `from .aligner import ConvAligner, MultiScaleAligner` 추가
- `__all__`에 `"ConvAligner"`, `"MultiScaleAligner"` 추가

### 3. `ultralytics/utils/loss.py` (수정)
파일 끝에 추가:

```python
class KDFeatureLoss:
    """Feature-level Knowledge Distillation loss. 기본: MSE."""

    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn or nn.MSELoss(reduction="mean")

    def __call__(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: list[Tensor] - aligned student features
            teacher_feats: list[Tensor] - teacher features
        Returns:
            scalar loss tensor
        """
        loss = sum(self.loss_fn(sf, tf.detach()) for sf, tf in zip(student_feats, teacher_feats))
        return loss / len(student_feats)
```

### 4. `ultralytics/cfg/default.yaml` (수정)
line 130 (`cfg:`) 근처에 추가:

```yaml
distill_cfg:  # (str, optional) path to distillation config YAML; enables KD when set
```

- 기본값 None → KD 비활성화
- 경로가 지정되면 해당 YAML을 로드하여 KD 활성화

### 5. `ultralytics/engine/distiller.py` (신규 - 핵심)

#### _FeatureHook 클래스
```python
class _FeatureHook:
    """Callable hook that appends layer output to a storage list.
    Lambda 대신 사용 — deepcopy/pickle 호환 (checkpoint 저장 시 필요)."""

    def __init__(self, storage):
        self.storage = storage

    def __call__(self, module, input, output):
        self.storage.append(output)
```

#### DistillationWrapper 클래스
```python
class DistillationWrapper(nn.Module):
    """Student + Aligner를 하나의 Module로 감싸서 build_optimizer가 둘 다 순회."""

    def __init__(self, student, aligner):
        super().__init__()
        self.student = student
        self.aligner = aligner

    def __getattr__(self, name):
        """student의 속성(stride, names, nc, yaml, args 등) 위임."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student, name)

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    def loss(self, batch, preds=None):
        return self.student.loss(batch, preds)

    def init_criterion(self):
        return self.student.init_criterion()
```

**build_optimizer 통합 원리**: `build_optimizer`는 `unwrap_model(model).named_modules()`를 순회 (`trainer.py:995`). DistillationWrapper는 nn.Module이므로 `named_modules()`가 `self.student`와 `self.aligner`의 모든 하위 모듈을 자동으로 포함. 코드 수정 불필요.

#### create_distiller 팩토리

```python
def create_distiller(trainer_cls):
    class Distiller(trainer_cls):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg, overrides, _callbacks)
            self.distill_cfg = self._load_distill_cfg()

        def _load_distill_cfg(self):
            """distill_cfg YAML 로드. None이면 KD 비활성화."""
            if not self.args.distill_cfg:
                return None
            return yaml_load(self.args.distill_cfg)

        def _setup_train(self):
            """KD 비활성화 시 부모 그대로, 활성화 시 teacher/aligner/hook 추가."""
            if self.distill_cfg is None:
                return super()._setup_train()

            # === KD 활성화 시 ===
            # 1. 부모의 setup_model()로 student 로딩
            ckpt = self.setup_model()
            self.model = self.model.to(self.device)
            self.set_model_attributes()

            # 2. distill_cfg에서 설정 읽기
            teacher_cfg = self.distill_cfg["teacher"]
            student_cfg = self.distill_cfg["student"]
            self.kd_weight = self.distill_cfg.get("weight", 1.0)

            # 3. Teacher 로딩 & freeze
            self._setup_teacher(teacher_cfg["model"])

            # 4. Distillation point 결정
            teacher_layers = teacher_cfg.get("layers") or self._get_head_input_indices(self.teacher)
            student_layers = student_cfg.get("layers") or self._get_head_input_indices(self.model)
            assert len(teacher_layers) == len(student_layers), "teacher/student layer 수 불일치"

            # 5. 레이어별 채널 수 추출 (더미 forward)
            teacher_channels = self._get_layer_channels(self.teacher, teacher_layers)
            student_channels = self._get_layer_channels(self.model, student_layers)

            # 6. Aligner 생성
            self._setup_aligner(student_channels, teacher_channels)

            # 7. Forward hook 등록
            self._teacher_feats = []
            self._student_feats = []
            self._hooks = []
            self._hooks += self._register_feature_hooks(self.teacher, teacher_layers, self._teacher_feats)
            self._hooks += self._register_feature_hooks(self.model, student_layers, self._student_feats)

            # 8. Wrapper로 감싸기
            self.model = DistillationWrapper(self.model, self.aligner_module)

            # 9. KD loss 초기화
            self._setup_kd_loss()

            # 10. 나머지 부모 _setup_train 로직
            #     (compile, freeze, AMP, DDP, batch, optimizer, scheduler, EMA 등)
            #     BaseTrainer._setup_train()의 line 301~362 로직 실행
            ...

        def _do_train(self, world_size=1):
            """부모의 _do_train 복사 + forward 부분에 KD 로직 삽입."""
            # 부모의 _do_train을 그대로 복사하되,
            # forward 블록(trainer.py:429-447)만 아래로 교체:

            # === KD Forward 블록 ===
            # self._teacher_feats.clear()
            # self._student_feats.clear()
            # with torch.no_grad():
            #     self.teacher(batch["img"])              # hook → teacher_feats 캡처
            # loss, self.loss_items = self.model(batch)   # hook → student_feats + task loss
            # aligned = unwrap_model(self.model).aligner(self._student_feats)
            # kd_loss = self.kd_loss_fn(aligned, self._teacher_feats)
            # self.loss = loss.sum() + self.kd_weight * kd_loss
            ...

        def save_model(self):
            """Student → best.pt/last.pt (표준 YOLO 호환), Aligner → aligner_last.pt."""
            # EMA에서 student만 추출하여 표준 체크포인트 저장
            # aligner를 self.wdir / "aligner_last.pt"로 별도 저장
            ...

        # === Helper 메서드 ===

        def _setup_teacher(self, model_path):
            """Teacher 로딩, device 이동, 전체 freeze, eval 모드."""
            weights, _ = load_checkpoint(model_path)
            self.teacher = weights.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        def _get_head_input_indices(self, model):
            """기본 distillation point: head의 from 인덱스."""
            return model.model[-1].f

        def _resolve_module(self, model, spec):
            """레이어 spec을 실제 nn.Module로 변환.
            정수 → model.model[spec], 문자열 → model.get_submodule(spec)."""
            if isinstance(spec, int):
                return model.model[spec]
            return model.get_submodule(spec)

        def _get_layer_channels(self, model, layer_specs):
            """더미 forward로 지정 레이어의 출력 채널 수를 측정."""
            channels = []
            temp_storage = []
            temp_hooks = []
            for spec in layer_specs:
                module = self._resolve_module(model, spec)
                h = module.register_forward_hook(
                    lambda m, inp, out, s=temp_storage: s.append(out)
                )
                temp_hooks.append(h)

            dummy = torch.zeros(1, 3, self.args.imgsz, self.args.imgsz, device=self.device)
            with torch.no_grad():
                model(dummy)

            for feat in temp_storage:
                channels.append(feat.shape[1])  # (B, C, H, W)

            for h in temp_hooks:
                h.remove()

            return channels

        def _register_feature_hooks(self, model, layer_specs, storage):
            """지정 레이어에 forward hook 등록. hook handle 리스트 반환."""
            handles = []
            for spec in layer_specs:
                module = self._resolve_module(model, spec)
                h = module.register_forward_hook(
                    lambda m, inp, out, s=storage: s.append(out)
                )
                handles.append(h)
            return handles

        def _setup_aligner(self, student_channels, teacher_channels):
            """MultiScaleAligner 생성."""
            aligner_name = self.distill_cfg.get("aligner", "ConvAligner")
            from ultralytics.nn.modules.aligner import ConvAligner, MultiScaleAligner
            aligner_map = {"ConvAligner": ConvAligner}
            self.aligner_module = MultiScaleAligner(
                student_channels, teacher_channels,
                aligner_cls=aligner_map[aligner_name],
            ).to(self.device)

        def _setup_kd_loss(self):
            """KDFeatureLoss 초기화."""
            loss_name = self.distill_cfg.get("loss", "mse")
            from ultralytics.utils.loss import KDFeatureLoss
            loss_map = {"mse": nn.MSELoss(reduction="mean")}
            self.kd_loss_fn = KDFeatureLoss(loss_fn=loss_map.get(loss_name))

    Distiller.__name__ = "Distiller"
    Distiller.__qualname__ = "Distiller"
    return Distiller
```

### 6. `ultralytics/engine/__init__.py` (수정)
- `from .distiller import create_distiller, DistillationWrapper` 추가

---

## 사용법 예시

```python
from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

Distiller = create_distiller(DetectionTrainer)
trainer = Distiller(overrides={
    "model": "yolov8n.yaml",
    "data": "coco8.yaml",
    "epochs": 2,
    "distill_cfg": "distill_cfg.yaml",
})
trainer.train()
```

---

## 핵심 학습 흐름

```
매 배치:
1. teacher_feats.clear(), student_feats.clear()
2. [no_grad] teacher(img) → hook이 teacher_feats에 6개 feature 캡처
3. student(batch) → hook이 student_feats에 6개 feature 캡처 + task_loss
4. aligned = aligner(student_feats)  # 6개 ConvAligner(1x1-ReLU-1x1) 적용
5. kd_loss = MSE(aligned, teacher_feats) / 6
6. total_loss = task_loss + kd_weight * kd_loss
7. backward → student + aligner gradient 업데이트
```

---

## TODO List (TDD)

### Phase 1: Aligner 모듈
- [x] **TODO-1**: `ultralytics/nn/modules/aligner.py` 생성 ✅
  - ConvAligner (1x1 Conv → ReLU → 1x1 Conv)
  - MultiScaleAligner (ModuleList로 여러 ConvAligner 관리)
  - **테스트**: shape 검증 — `ConvAligner(64, 80)` 입력 `(1,64,20,20)` → 출력 `(1,80,20,20)` ✅
  - **테스트**: MultiScaleAligner([64,64,64], [80,80,80]) 6개 feature 처리 ✅

- [x] **TODO-2**: `ultralytics/nn/modules/__init__.py` 수정 ✅
  - ConvAligner, MultiScaleAligner import/export 추가
  - **테스트**: `from ultralytics.nn.modules import ConvAligner, MultiScaleAligner` 정상 import ✅
  - **테스트 파일**: `tests/test_kd_phase1.py`

### Phase 2: KD Loss
- [x] **TODO-3**: `ultralytics/utils/loss.py`에 KDFeatureLoss 추가 ✅
  - 기본 MSE, distillation point별 평균
  - **테스트**: 동일 tensor 시 loss≈0, 다른 tensor 시 loss>0 ✅
  - **테스트**: teacher_feats가 .detach() 처리되는지 확인 ✅
  - **테스트 파일**: `tests/test_kd_phase2.py`

### Phase 3: Config 설정
- [x] **TODO-4**: `ultralytics/cfg/default.yaml`에 `distill_cfg:` 키 추가 ✅
  - 기본값 None → KD 비활성화
  - **테스트**: `from ultralytics.cfg import DEFAULT_CFG_DICT; assert "distill_cfg" in DEFAULT_CFG_DICT` ✅

- [x] **TODO-4.1**: `ultralytics/cfg/distill_cfg.yaml` 예시 파일 생성 ✅
  - YOLOv8 head cv2/cv3 2번째 conv 증류 설정 (6 points)
  - teacher: yolov8s.pt, aligner: ConvAligner, loss: mse, weight: 1.0
  - **테스트**: YAML 로드 → teacher.model, teacher.layers, student.layers 키 존재 확인 ✅
  - **테스트 파일**: `tests/test_kd_phase3.py`

### Phase 4: Distiller 엔진 (핵심)
- [x] **TODO-5**: `ultralytics/engine/distiller.py` — DistillationWrapper 구현 ✅
  - student + aligner를 nn.Module로 감싸기
  - student 속성 위임 (__getattr__), forward/loss/init_criterion 위임
  - **테스트**: wrapper.stride == student.stride, wrapper.names == student.names ✅
  - **테스트**: wrapper.named_modules()에 student + aligner 모듈 모두 포함 ✅

- [x] **TODO-6**: create_distiller 팩토리 — distill_cfg 로딩 ✅
  - _load_distill_cfg(): YAML → dict, None이면 KD off
  - **테스트**: distill_cfg 지정 시 정상 로드, 미지정 시 None ✅

- [x] **TODO-7**: _setup_teacher — Teacher 로딩 & freeze ✅
  - load_checkpoint → device → freeze → eval
  - **테스트**: teacher.parameters() 전부 requires_grad=False ✅

- [x] **TODO-8**: _resolve_module + _register_feature_hooks ✅
  - 정수 → model.model[idx], 문자열 → model.get_submodule(path)
  - **테스트**: yolov8n "model.22.cv2.0.1" hook에서 feature 캡처 확인 ✅
  - **테스트**: 정수 인덱스 15에서 module resolve 확인 ✅

- [x] **TODO-9**: _get_layer_channels — 더미 forward로 채널 측정 ✅
  - **테스트**: yolov8n cv2[0][1]=64, cv3[0][1]=80 채널 확인 ✅

- [x] **TODO-10**: _setup_aligner + Wrapper 조립 ✅
  - MultiScaleAligner(student_ch, teacher_ch) → DistillationWrapper
  - **테스트**: aligner 개수 == distillation point 개수 (6개) ✅

- [x] **TODO-11**: build_optimizer 통합 검증 ✅
  - build_optimizer(wrapper)가 student + aligner 파라미터 모두 포함
  - **테스트**: optimizer param_groups에 aligner 파라미터 존재 ✅
  - **테스트 파일**: `tests/test_kd_phase4.py`

- [x] **TODO-12**: _setup_train 오버라이드 완성 ✅
  - teacher → aligner → hook → wrapper → compile/freeze/AMP/DDP/optimizer/EMA
  - **테스트**: setup 후 self.model이 DistillationWrapper, self.teacher가 eval 모드 ✅

- [x] **TODO-13**: _do_train 오버라이드 — KD forward 로직 ✅
  - teacher forward(no_grad) → student forward → align → KD loss → total loss
  - **테스트**: coco8 1 epoch 학습 → loss 값 출력 (kd_loss 포함) ✅
  - 참고: hook lambda → _FeatureHook 클래스로 교체 (pickle 호환)

- [x] **TODO-14**: save_model 오버라이드 ✅
  - best.pt/last.pt: student만 (표준 YOLO 호환)
  - aligner_last.pt: aligner 별도 저장
  - **테스트**: last.pt를 `YOLO("last.pt")` 로 로드 가능 ✅
  - **테스트**: aligner_last.pt에 aligner 키 존재 ✅
  - **테스트 파일**: `tests/test_kd_phase4.py`, `tests/test_kd_phase4_train.py`

### Phase 5: Export & 통합
- [x] **TODO-15**: `ultralytics/engine/__init__.py` 수정 ✅
  - create_distiller, DistillationWrapper export 추가
  - **테스트**: `from ultralytics.engine import create_distiller` ✅
  - **테스트 파일**: `tests/test_kd_phase5.py`

### Phase 6: End-to-End 검증
- [ ] **TODO-16**: 전체 distillation 학습 테스트 (보류 - 다음 세션에서 진행)
  - Teacher: yolov8s.pt, Student: yolov8n.yaml, Data: VOC.yaml
  - distill_cfg.yaml: cv2/cv3 2번째 conv 증류 (6 points)
  - 5 epoch 학습 → checkpoint 저장 → student 추론
  - workers=4로 설정 (메모리 이슈 방지)
  - venv 활성화 필요: `source venv/Scripts/activate` (CUDA torch 설치됨)
  - **테스트**: 학습 완료, teacher frozen, checkpoint 호환, 추론 동작
  - **테스트 파일**: `tests/test_kd_phase6.py` (workers=4 추가됨)
