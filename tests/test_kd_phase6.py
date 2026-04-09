# python -m pytest tests/test_kd_phase6.py -v
"""Phase 6: End-to-End 검증 — Baseline vs KD 결과 확인 테스트."""

from pathlib import Path

import torch
import pytest

BASELINE_DIR = Path("runs/detect/baseline_yolov8n")
KD_DIR = Path("runs/detect/kd_yolov8n")


# ── TODO-16: Baseline 결과 검증 ──────────────────────────────────────────────


class TestBaseline:
    """TODO-16: Baseline (yolov8n.pt, VOC, 50 epochs) 학습 결과 검증."""

    def test_checkpoints_exist(self):
        """best.pt, last.pt 체크포인트 존재 확인."""
        assert (BASELINE_DIR / "weights" / "best.pt").exists()
        assert (BASELINE_DIR / "weights" / "last.pt").exists()

    def test_results_csv_exists(self):
        """results.csv 존재 및 50 epoch 기록 확인."""
        csv_path = BASELINE_DIR / "results.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 51, f"Expected 51+ lines (header + 50 epochs), got {len(lines)}"

    def test_best_pt_loads(self):
        """best.pt를 YOLO로 정상 로드 가능 확인."""
        from ultralytics import YOLO

        model = YOLO(str(BASELINE_DIR / "weights" / "best.pt"))
        assert model is not None
        assert hasattr(model, "predict")

    def test_best_pt_inference(self):
        """best.pt로 추론 동작 확인."""
        from ultralytics import YOLO

        model = YOLO(str(BASELINE_DIR / "weights" / "best.pt"))
        results = model.predict("ultralytics/assets/bus.jpg", verbose=False)
        assert len(results) == 1
        assert len(results[0].boxes) > 0, "No detections on bus.jpg"


# ── TODO-17: KD 결과 검증 ────────────────────────────────────────────────────


class TestKD:
    """TODO-17: KD (yolov8n.pt + yolov8s teacher, VOC, 50 epochs) 학습 결과 검증."""

    def test_checkpoints_exist(self):
        """best.pt, last.pt, aligner_last.pt 체크포인트 존재 확인."""
        assert (KD_DIR / "weights" / "best.pt").exists()
        assert (KD_DIR / "weights" / "last.pt").exists()
        assert (KD_DIR / "weights" / "aligner_last.pt").exists()

    def test_results_csv_exists(self):
        """results.csv 존재 및 50 epoch 기록 확인."""
        csv_path = KD_DIR / "results.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 51, f"Expected 51+ lines (header + 50 epochs), got {len(lines)}"

    def test_results_csv_has_kd_loss(self):
        """results.csv에 train/kd_loss 컬럼 존재 확인."""
        csv_path = KD_DIR / "results.csv"
        header = csv_path.read_text().strip().split("\n")[0]
        assert "kd_loss" in header, f"kd_loss column not found in header: {header}"

    def test_best_pt_loads_as_standard_yolo(self):
        """best.pt가 표준 YOLO 모델로 로드 (DistillationWrapper 아님) 확인."""
        from ultralytics import YOLO
        from ultralytics.engine.distiller import DistillationWrapper

        model = YOLO(str(KD_DIR / "weights" / "best.pt"))
        assert not isinstance(model.model, DistillationWrapper)

    def test_best_pt_inference(self):
        """best.pt로 추론 동작 확인."""
        from ultralytics import YOLO

        model = YOLO(str(KD_DIR / "weights" / "best.pt"))
        results = model.predict("ultralytics/assets/bus.jpg", verbose=False)
        assert len(results) == 1
        assert len(results[0].boxes) > 0, "No detections on bus.jpg"

    def test_aligner_checkpoint(self):
        """aligner_last.pt에 aligner 키 존재 확인."""
        ckpt = torch.load(str(KD_DIR / "weights" / "aligner_last.pt"), map_location="cpu", weights_only=False)
        assert "aligner" in ckpt, f"Keys found: {list(ckpt.keys())}"


# ── TODO-18: 성능 비교 ───────────────────────────────────────────────────────


class TestComparison:
    """TODO-18: Baseline vs KD 성능 비교."""

    @staticmethod
    def _get_best_metrics(csv_path):
        """results.csv에서 best mAP50 epoch의 metrics 반환."""
        lines = csv_path.read_text().strip().split("\n")
        header = [h.strip() for h in lines[0].split(",")]
        best_map50 = -1
        best_row = None
        map50_idx = next(i for i, h in enumerate(header) if "mAP50(B)" in h and "mAP50-95" not in h)
        for line in lines[1:]:
            vals = [v.strip() for v in line.split(",")]
            m = float(vals[map50_idx])
            if m > best_map50:
                best_map50 = m
                best_row = vals
        return dict(zip(header, best_row))

    def test_both_results_exist(self):
        """두 학습 결과 모두 존재 확인."""
        assert (BASELINE_DIR / "results.csv").exists(), "Baseline results missing"
        assert (KD_DIR / "results.csv").exists(), "KD results missing"

    @staticmethod
    def _get_final_metrics(csv_path):
        """results.csv에서 마지막 epoch의 metrics 반환."""
        lines = csv_path.read_text().strip().split("\n")
        header = [h.strip() for h in lines[0].split(",")]
        vals = [v.strip() for v in lines[-1].split(",")]
        return dict(zip(header, vals))

    @staticmethod
    def _get_training_time(csv_path):
        """results.csv에서 총 학습 시간(초) 반환 (마지막 epoch의 time 컬럼)."""
        lines = csv_path.read_text().strip().split("\n")
        header = [h.strip() for h in lines[0].split(",")]
        time_idx = header.index("time")
        last_vals = [v.strip() for v in lines[-1].split(",")]
        return float(last_vals[time_idx])

    @staticmethod
    def _get_model_size_mb(pt_path):
        """체크포인트 파일 크기(MB) 반환."""
        return pt_path.stat().st_size / (1024 * 1024)

    def test_kd_improves_map50(self):
        """KD 모델의 best mAP50이 baseline보다 높은지 검증."""
        baseline = self._get_best_metrics(BASELINE_DIR / "results.csv")
        kd = self._get_best_metrics(KD_DIR / "results.csv")

        def val(d, key_substr):
            for k, v in d.items():
                if key_substr in k:
                    return float(v)
            return 0.0

        b_map50 = val(baseline, "mAP50(B)")
        k_map50 = val(kd, "mAP50(B)")
        assert k_map50 > b_map50, f"KD mAP50 ({k_map50:.4f}) should be > baseline ({b_map50:.4f})"

    def test_kd_improves_map50_95(self):
        """KD 모델의 best mAP50-95가 baseline보다 높은지 검증."""
        baseline = self._get_best_metrics(BASELINE_DIR / "results.csv")
        kd = self._get_best_metrics(KD_DIR / "results.csv")

        def val(d, key_substr):
            for k, v in d.items():
                if key_substr in k:
                    return float(v)
            return 0.0

        b_map50_95 = val(baseline, "mAP50-95")
        k_map50_95 = val(kd, "mAP50-95")
        assert k_map50_95 > b_map50_95, f"KD mAP50-95 ({k_map50_95:.4f}) should be > baseline ({b_map50_95:.4f})"

    def test_print_comparison(self):
        """성능 비교표 출력 (항상 통과, 비교 목적)."""
        baseline = self._get_best_metrics(BASELINE_DIR / "results.csv")
        kd = self._get_best_metrics(KD_DIR / "results.csv")

        def val(d, key_substr):
            for k, v in d.items():
                if key_substr in k:
                    return float(v)
            return 0.0

        b_map50 = val(baseline, "mAP50(B)")
        b_map50_95 = val(baseline, "mAP50-95")
        k_map50 = val(kd, "mAP50(B)")
        k_map50_95 = val(kd, "mAP50-95")

        b_time = self._get_training_time(BASELINE_DIR / "results.csv")
        k_time = self._get_training_time(KD_DIR / "results.csv")

        b_size = self._get_model_size_mb(BASELINE_DIR / "weights" / "best.pt")
        k_size = self._get_model_size_mb(KD_DIR / "weights" / "best.pt")

        print("\n" + "=" * 70)
        print("  Baseline vs KD  —  Performance Comparison (VOC, 50 epochs)")
        print("=" * 70)
        print(f"  {'Metric':<20} {'Baseline':>14} {'KD':>14} {'Diff':>14}")
        print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14}")
        print(f"  {'mAP50':<20} {b_map50:>14.4f} {k_map50:>14.4f} {k_map50 - b_map50:>+14.4f}")
        print(f"  {'mAP50-95':<20} {b_map50_95:>14.4f} {k_map50_95:>14.4f} {k_map50_95 - b_map50_95:>+14.4f}")
        print(f"  {'Training Time (s)':<20} {b_time:>14.1f} {k_time:>14.1f} {k_time - b_time:>+14.1f}")
        print(f"  {'Model Size (MB)':<20} {b_size:>14.2f} {k_size:>14.2f} {k_size - b_size:>+14.2f}")
        print("=" * 70)
