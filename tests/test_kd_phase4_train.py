"""Phase 4 tests: TODO-12 (_setup_train), TODO-13 (_do_train), TODO-14 (save_model).

Runs actual 1-epoch distillation training on coco8 to verify the full pipeline.
"""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DISTILL_CFG = ROOT / "ultralytics" / "cfg" / "distill_cfg.yaml"


def test_setup_train_and_train_1_epoch():
    """TODO-12/13/14: Full distillation pipeline - setup, 1 epoch train, checkpoint save."""
    from ultralytics.engine.distiller import DistillationWrapper, create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "epochs": 1,
        "batch": 4,
        "imgsz": 320,
        "device": "cpu",
        "distill_cfg": str(DISTILL_CFG),
        "workers": 0,
        "val": False,
        "plots": False,
    })

    # Run training (1 epoch)
    trainer.train()

    # TODO-12: Verify _setup_train results
    from ultralytics.utils.torch_utils import unwrap_model
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationWrapper), f"Expected DistillationWrapper, got {type(model)}"
    assert hasattr(trainer, "teacher"), "Missing teacher attribute"
    assert not trainer.teacher.training, "Teacher should be in eval mode"
    print("TODO-12: _setup_train ✓")

    # TODO-13: Verify training ran (loss exists)
    assert trainer.loss is not None, "Loss should not be None after training"
    print(f"TODO-13: _do_train completed, final loss = {trainer.loss.item():.4f} ✓")

    # TODO-14: Verify checkpoints
    save_dir = trainer.save_dir
    wdir = save_dir / "weights"
    last_pt = wdir / "last.pt"
    aligner_pt = wdir / "aligner_last.pt"

    assert last_pt.exists(), f"last.pt not found at {last_pt}"
    assert aligner_pt.exists(), f"aligner_last.pt not found at {aligner_pt}"

    # Verify last.pt is a standard YOLO checkpoint (student only, no wrapper)
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    ema_model = ckpt["ema"]
    assert not isinstance(ema_model, DistillationWrapper), "Checkpoint should contain student only, not wrapper"
    print(f"TODO-14: last.pt contains student model (no wrapper) ✓")

    # Verify can load as standard YOLO model
    from ultralytics import YOLO
    loaded = YOLO(str(last_pt))
    assert loaded.model is not None, "Failed to load checkpoint as YOLO model"
    print(f"TODO-14: YOLO(last.pt) loads successfully ✓")

    # Verify aligner checkpoint
    aligner_ckpt = torch.load(aligner_pt, map_location="cpu", weights_only=False)
    assert "aligner" in aligner_ckpt, "aligner_last.pt missing 'aligner' key"
    print(f"TODO-14: aligner_last.pt saved ✓")

    # Cleanup
    import shutil
    shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Phase 4 Training Tests (TODO-12/13/14) ===\n")
    test_setup_train_and_train_1_epoch()
    print("\nAll Phase 4 training tests passed!")
