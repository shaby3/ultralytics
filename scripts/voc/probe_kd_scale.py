# python scripts/voc/probe_kd_scale.py
"""Phase 3 프로브 — 기법별 1에폭 kd_loss 를 재서 weight 를 정한다.

기법마다 kd_loss 의 자릿수가 다르다. weight 를 그대로 두면 "어느 기법이 나은가"가 아니라
"어느 기법에 KD 신호를 많이 줬는가" 비교가 된다 — 위치 스윕에서 19배 차이로 겪은 문제다 (README §4).
그래서 위치 스윕과 같은 규칙을 쓴다: **MSE 런의 초기 KD 비중에 나머지를 맞춘다.**

    w = (kd_mse / task_mse) * task_method / kd_method        (w_mse = 1)

기준값을 과거 기록(5.5596)이 아니라 **이 프로브의 mse 런**에서 새로 뽑는다. 같은 세션·같은 조건이라
비교가 더 깨끗하고, 동시에 회귀 검사가 된다 — 리팩터가 기존 경로를 안 건드렸다면 mse 런의
ep1 kd_loss 가 5.56 근처로 재현되어야 한다.

kd_loss 는 배치 평균이라 batch 크기와 무관하다. 트레이너는 **weight 를 곱한 값**을 기록하므로
(distiller.py 의 `kd_val = ... * self.kd_weight`) 여기서 config 의 weight 로 되나눠 원값으로 만든다 —
weight 산출 공식은 원값 기준이다. (예전에는 원값이 기록됐고 JSON 의 기존 항목들도 원값이다.)

결과는 매 런마다 JSON 에 적고, 이미 잰 항목은 건너뛴다. 중간에 죽어도 다시 실행하면 이어붙는다.
학습 산출물은 runs/ 를 오염시키지 않도록 scratch 아래에 둔다.

각 런 약 8분, 3런 25분.
"""

import json
import time
from pathlib import Path

import torch

from ultralytics.engine.distiller import create_distiller
from ultralytics.models.yolo.detect.train import DetectionTrainer

# runs/ 밖에 둔다 — 본 실험 결과가 아니라 설정을 정하기 위한 측정이다
SCRATCH = Path.home() / "AppData/Local/Temp/claude/kd_probe"
OUT = Path(__file__).parent / "probe_kd_scale.json"

CONFIGS = {
    "mse": "ultralytics/cfg/distill_head1_from_8s_coco.yaml",  # 기준 + 회귀 검사
    "pkd": "ultralytics/cfg/distill_head1_from_8s_coco_pkd.yaml",
    "mgd": "ultralytics/cfg/distill_head1_from_8s_coco_mgd.yaml",
    "fgd": "ultralytics/cfg/distill_head1_from_8s_coco_fgd.yaml",
}


def probe(tag, cfg):
    """1에폭 돌리고 ep1 의 kd_loss·task loss·peak VRAM·소요시간을 돌려준다."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(
        overrides={
            "model": "yolov8n.pt",
            "data": "VOC.yaml",
            "project": str(SCRATCH / tag),
            "name": "train",
            "epochs": 1,
            "batch": 16,  # 본 런과 같아야 VRAM·시간 실측이 의미가 있다
            "imgsz": 640,
            "workers": 2,
            "exist_ok": True,
            "device": 0,
            "amp": True,
            "plots": False,
            "val": False,  # 검증은 이 측정에 불필요하다
            "distill_cfg": cfg,
        }
    )
    trainer.train()

    # 트레이너가 에폭 평균으로 누적해 둔 값을 그대로 읽는다 (results.csv 파싱보다 확실하다).
    # 기록값은 weight 가 곱해져 있으므로 config 의 weight 로 되나눠 원값으로 만든다.
    from ultralytics.utils import YAML

    kd = float(trainer.tkd_loss) / YAML.load(cfg).get("weight", 1.0)
    task = float(sum(trainer.tloss))  # box + cls + dfl
    return {
        "kd_loss": kd,
        "task_loss": task,
        "kd_share": kd / (kd + task),  # weight=1.0 기준 초기 KD 비중
        "peak_vram_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "minutes": (time.time() - t0) / 60,
        "config": cfg,
    }


if __name__ == "__main__":
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    for tag, cfg in CONFIGS.items():
        if tag in results:
            print(f"[skip] {tag} — 이미 측정됨 (kd_loss={results[tag]['kd_loss']:.4f})")
            continue
        print(f"[run ] {tag} — {cfg}")
        results[tag] = probe(tag, cfg)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")  # 런마다 즉시 저장
        print(f"[done] {tag} — {results[tag]}")

    # --- weight 산출 ---
    ref = results["mse"]
    ratio = ref["kd_loss"] / ref["task_loss"]  # MSE 가 weight 1.0 에서 갖는 KD:task 비

    print(f"\n{'기법':<6}{'kd_loss':>12}{'task':>10}{'KD비중':>9}{'권장 weight':>13}{'peak GB':>10}{'분/에폭':>9}")
    for tag, r in results.items():
        w = ratio * r["task_loss"] / r["kd_loss"]
        share = w * r["kd_loss"] / (w * r["kd_loss"] + r["task_loss"])
        print(
            f"{tag:<6}{r['kd_loss']:>12.4f}{r['task_loss']:>10.3f}{r['kd_share']:>8.1%}"
            f"{w:>13.4f}{r['peak_vram_gb']:>10.2f}{r['minutes']:>9.1f}   (보정 후 KD비중 {share:.1%})"
        )

    print(f"\n회귀 검사 — mse 의 ep1 kd_loss {ref['kd_loss']:.4f} (본 런 기록 5.5596 근처여야 한다)")
