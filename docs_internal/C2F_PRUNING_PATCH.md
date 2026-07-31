# C2f 프루닝용 변경 (cv0/cv1 분리) — 복원 가이드

## 배경

구조적 프루닝 도구(torch-pruning 등)는 `chunk()` / `split()`으로 채널을 나누는 연산의
의존성 그래프를 제대로 추적하지 못한다. upstream YOLOv8의 `C2f`는 `cv1`이 `2*c` 채널을
한 번에 출력한 뒤 `chunk(2, 1)`으로 쪼개기 때문에, 프루닝 시 두 분기의 채널 대응이 깨진다.

이를 피하기 위해 `cv1`을 **`cv0` + `cv1` 두 개의 독립 Conv**로 분리하는 패치를 적용했었다
(커밋 `ba2423c4`).

## ⚠️ 이 패치의 부작용 — 사전학습 가중치 유실

배포된 COCO 사전학습 `.pt`는 upstream 구조(`cv1` 2c 채널)를 따르므로, 패치 상태에서는
state_dict 키가 매칭되지 않아 C2f 가중치가 **조용히 유실**된다. 실측 결과:

| 상태 | yolov8n 가중치 전이 |
|------|--------------------|
| 패치 적용 (cv0/cv1 분리) | `Transferred 315/403 items` ← 88개 유실 |
| upstream 원복 | `Transferred 355/355 items` ✅ |

C2f는 YOLOv8 backbone/neck의 핵심 블록이라, 유실 상태로 학습하면 사전학습 효과를
상당 부분 잃는다. **따라서 baseline·KD 실험 중에는 upstream 원본을 유지한다.**

패치 상태에서 사전학습 가중치를 쓰려면 `convert_weights.py`로 체크포인트를 먼저 변환해야 한다
(legacy `cv1`(2c) → `cv0`(앞 절반) / `cv1`(뒤 절반) 분리).

## 현재 상태

**upstream 원본으로 원복됨** (`ultralytics/nn/modules/block.py`의 `C2f`).

## 프루닝이 필요해질 때 — 패치 재적용 방법

`ultralytics/nn/modules/block.py`의 `C2f`를 아래로 교체한다.

### `__init__` — cv1을 cv0/cv1로 분리
```python
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```

### `forward` — chunk 제거
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

### `forward_split` — legacy 체크포인트 호환 분기 포함
```python
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        if hasattr(self, "cv0"):
            y = [self.cv0(x), self.cv1(x)]
        else:
            y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

### 재적용 후 필수 절차
1. 사용할 사전학습/체크포인트를 변환:
   ```bash
   python convert_weights.py yolov8n.pt yolov8n_c2f.pt yolov8n.yaml
   ```
2. 학습 로그의 `Transferred X/Y items`가 전량 전이인지 확인 (부분 전이면 변환 누락).

## 참고
- upstream 원본 코드는 git에서 확인 가능: `git show ba2423c4^:ultralytics/nn/modules/block.py`
- 변환 스크립트: `convert_weights.py` (C2f 상속 블록 C3k2/C2fCIB/C2fPSA 등도 자동 처리)
