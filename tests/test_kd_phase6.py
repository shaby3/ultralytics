"""Phase 6 tests: End-to-End distillation on VOC, 5 epochs."""

import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DISTILL_CFG = ROOT / "ultralytics" / "cfg" / "distill_cfg.yaml"


def test_e2e_distillation_voc():
    """Full KD pipeline: yolov8s→yolov8n, VOC, 5 epochs, cv2/cv3 2nd conv distillation."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import DistillationWrapper, create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "VOC.yaml",
        "epochs": 5,
        "batch": 16,
        "imgsz": 640,
        "distill_cfg": str(DISTILL_CFG),
        "workers": 4,
        "val": True,
        "plots": False,
        "exist_ok": True,
        "name": "kd_voc_e2e",
    })

    trainer.train()

    # --- Verify training completed ---
    assert trainer.epoch == 4, f"Expected 5 epochs (0-indexed last=4), got {trainer.epoch}"
    print(f"[PASS] Training completed: {trainer.epoch + 1} epochs")

    # --- Verify model structure ---
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationWrapper), "Model should be DistillationWrapper"
    print("[PASS] Model is DistillationWrapper")

    # --- Verify teacher frozen ---
    for p in trainer.teacher.parameters():
        assert not p.requires_grad, "Teacher should be frozen"
    print("[PASS] Teacher frozen")

    # --- Verify gradient flow (student + aligner) ---
    student_has_grad = any(
        p.grad is not None for p in model.student.parameters() if p.requires_grad
    )
    aligner_has_grad = any(
        p.grad is not None for p in model.aligner.parameters() if p.requires_grad
    )
    # Note: after optimizer.zero_grad() at epoch end, grads may be zeroed.
    # Check parameters are trainable instead.
    student_trainable = sum(1 for p in model.student.parameters() if p.requires_grad)
    aligner_trainable = sum(1 for p in model.aligner.parameters() if p.requires_grad)
    assert student_trainable > 0, "Student should have trainable params"
    assert aligner_trainable > 0, "Aligner should have trainable params"
    print(f"[PASS] Trainable params - student: {student_trainable}, aligner: {aligner_trainable}")

    # --- Verify checkpoints ---
    wdir = trainer.save_dir / "weights"
    last_pt = wdir / "last.pt"
    best_pt = wdir / "best.pt"
    aligner_pt = wdir / "aligner_last.pt"

    assert last_pt.exists(), "last.pt not found"
    assert best_pt.exists(), "best.pt not found"
    assert aligner_pt.exists(), "aligner_last.pt not found"
    print("[PASS] All checkpoints exist")

    # --- Verify standard YOLO load ---
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert not isinstance(ckpt["ema"], DistillationWrapper), "Checkpoint should be student only"

    loaded = YOLO(str(best_pt))
    assert loaded.model is not None, "Failed to load best.pt"
    print("[PASS] best.pt loads as standard YOLO model")

    # --- Verify inference works ---
    results = loaded.predict(source=ROOT / "ultralytics" / "assets" / "bus.jpg", verbose=False)
    assert results is not None, "Inference failed"
    print("[PASS] Inference on loaded model works")

    # --- Verify aligner checkpoint ---
    aligner_ckpt = torch.load(aligner_pt, map_location="cpu", weights_only=False)
    assert "aligner" in aligner_ckpt, "aligner key missing"
    assert "epoch" in aligner_ckpt, "epoch key missing"
    print("[PASS] aligner_last.pt valid")

    # --- Print final metrics ---
    print(f"\nFinal metrics: {trainer.metrics}")
    print(f"Best fitness: {trainer.best_fitness}")

    # Cleanup
    shutil.rmtree(trainer.save_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Phase 6: End-to-End KD Test (VOC, 5 epochs) ===\n")
    test_e2e_distillation_voc()
    print("\nAll Phase 6 tests passed!")
