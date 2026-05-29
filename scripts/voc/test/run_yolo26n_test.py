"""VOC baseline test — yolo26n best.pt evaluated on test split (test2007)."""

import csv
import os

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

if __name__ == "__main__":
    model_path = "runs/detect/voc/train/baseline_yolo26n_100epoch/weights/best.pt"
    model = YOLO(model_path)

    model_size_mb = os.path.getsize(model_path) / 1e6
    n_params = get_num_params(model.model)
    flops = get_flops(model.model, imgsz=640)

    results = model.val(
        data="VOC.yaml",
        split="test",
        imgsz=640,
        batch=1,
        workers=2,
        project="voc/test",
        name="baseline_yolo26n_100epoch_test",
        exist_ok=True,
        device=0,
    )

    inference_ms = results.speed["inference"]
    metrics = {
        "model_path": model_path,
        "params_M":   round(n_params / 1e6, 3),
        "GFLOPs":     round(flops, 1),
        "size_MB":    round(model_size_mb, 1),
        "mAP50":      round(results.results_dict["metrics/mAP50(B)"], 4),
        "mAP50_95":   round(results.results_dict["metrics/mAP50-95(B)"], 4),
        "latency_ms": round(inference_ms, 2),
        "FPS":        round(1000 / inference_ms, 1),
    }

    csv_path = results.save_dir / "metrics_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics saved → {csv_path}")
