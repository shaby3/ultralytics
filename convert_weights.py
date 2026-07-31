"""
legacy checkpoint → 현재 C2f (cv0/cv1 분리) 구조로 변환.

C2f를 상속하는 모든 블록(C3k2, C2fCIB, C2fPSA 등)도 자동으로 처리됨.
nc(클래스 수)는 checkpoint에서 자동으로 감지하므로 yaml 인자와 달라도 안전하게 처리됨.

사용법:
    python convert_weights.py yolov8n.pt
        → yolov8n_c2f.pt (yaml은 yolov8n.yaml로 추론)

    python convert_weights.py best.pt out.pt
        → out.pt (yaml은 best.yaml로 추론 — 이름으로 추론 불가 시 3번째 인자 필요)

    python convert_weights.py best.pt best_converted.pt yolov8n.yaml
        → yaml 명시. nc 불일치 시 checkpoint의 nc로 자동 재빌드됨.

예시:
    python convert_weights.py \\
        runs/detect/voc_baseline_yolov8n/weights/best.pt \\
        runs/detect/voc_baseline_yolov8n/weights/best_converted.pt \\
        yolov8n.yaml
"""
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.patches import torch_load

BN_KEYS = ['bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']


def _detect_nc(model_obj) -> int | None:
    """checkpoint 모델 객체에서 nc를 읽는다. DetectionModel.nc → Detect.nc 순으로 fallback."""
    nc = getattr(model_obj, 'nc', None)
    if nc is None:
        for mod in model_obj.modules():
            if isinstance(mod, Detect):
                nc = mod.nc
                break
    return nc


def convert(src: str, dst: str, yaml_name: str | None = None) -> None:
    ckpt = torch_load(src, map_location='cpu')
    old_model_obj = ckpt.get('ema') or ckpt.get('model')
    old_state = old_model_obj.state_dict()

    if any('cv0' in k for k in old_state):
        print(f"{src} already has cv0 keys, skipping.")
        return

    if yaml_name is None:
        yaml_name = Path(src).stem + '.yaml'

    model = YOLO(yaml_name)

    # checkpoint의 nc와 yaml의 nc가 다르면 분류 head shape가 달라져 가중치 복사가 안 됨.
    # 체크포인트에 저장된 yaml(nc·scale 모두 정확)로 DetectionModel을 직접 재빌드한다.
    old_nc = _detect_nc(old_model_obj)
    if old_nc is not None and _detect_nc(model.model) != old_nc:
        print(f"[warn] nc 불일치 (yaml={_detect_nc(model.model)}, checkpoint={old_nc}) → nc={old_nc}로 재빌드")
        src_cfg = getattr(old_model_obj, 'yaml', None)
        if src_cfg is not None:
            model.model = DetectionModel(src_cfg, nc=old_nc, ch=src_cfg.get('channels', 3))
        else:
            raise RuntimeError("nc 불일치인데 checkpoint에 yaml이 없어 재빌드 불가. yaml_name을 nc가 맞는 yaml로 지정하세요.")

    new_state = model.model.state_dict()

    # C2f 및 그 서브클래스(C3k2, C2fCIB, C2fPSA 등)를 모두 처리
    # isinstance(module, C2f)는 상속 계층 전체를 커버함
    handled_prefixes = set()
    for name, module in model.model.named_modules():
        if not isinstance(module, C2f):
            continue
        prefix = name + '.'

        old_cv1_w_key = f'{prefix}cv1.conv.weight'
        if old_cv1_w_key not in old_state:
            continue

        # step [1] 처리 성공 확정 후 등록 (early continue 이전에 add하면 step [2]에서 오작동)
        handled_prefixes.add(prefix)

        old_w = old_state[old_cv1_w_key]
        half = old_w.shape[0] // 2

        # 구버전 cv1 (2*c 채널)을 cv0 (앞 절반) / cv1 (뒤 절반)으로 분리
        new_state[f'{prefix}cv0.conv.weight'] = old_w[:half]
        new_state[f'{prefix}cv1.conv.weight'] = old_w[half:]

        for bn_key in BN_KEYS:
            old_bn_k = f'{prefix}cv1.{bn_key}'
            if old_bn_k not in old_state:
                continue
            old_bn = old_state[old_bn_k]
            new_state[f'{prefix}cv0.{bn_key}'] = old_bn[:half]
            new_state[f'{prefix}cv1.{bn_key}'] = old_bn[half:]

        # num_batches_tracked는 스칼라라 split 불필요 — 양쪽에 동일하게 복사
        num_tracked_key = f'{prefix}cv1.bn.num_batches_tracked'
        if num_tracked_key in old_state:
            val = old_state[num_tracked_key]
            new_state[f'{prefix}cv0.bn.num_batches_tracked'] = val.clone()
            new_state[f'{prefix}cv1.bn.num_batches_tracked'] = val.clone()

    # [2] 나머지 레이어 (C2f 내부 포함): shape 일치 시 그대로 복사
    for k in list(new_state.keys()):
        # C2f 직속 cv0/cv1은 이미 위에서 처리 완료
        if any(k.startswith(p) for p in handled_prefixes):
            relative = k[len(next(p for p in handled_prefixes if k.startswith(p))):]
            if relative.startswith('cv0.') or relative.startswith('cv1.'):
                continue
        if k in old_state and old_state[k].shape == new_state[k].shape:
            new_state[k] = old_state[k]

    model.model.load_state_dict(new_state, strict=False)
    model.save(dst)
    print(f"Saved: {dst}")


if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.pt', '_c2f.pt')
    yaml_arg = sys.argv[3] if len(sys.argv) > 3 else None
    src = attempt_download_asset(src)
    convert(src, dst, yaml_arg)
