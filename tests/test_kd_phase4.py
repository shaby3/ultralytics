"""Phase 4 tests: Distiller engine (TODO-5 through TODO-14)."""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTILL_CFG = ROOT / "ultralytics" / "cfg" / "distill_cfg.yaml"


# === TODO-5: DistillationWrapper ===

def test_wrapper_attribute_delegation():
    """Wrapper delegates stride, names, nc to student."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import DistillationWrapper
    from ultralytics.nn.modules.aligner import MultiScaleAligner

    model = YOLO("yolov8n.pt").model
    aligner = MultiScaleAligner([64, 64], [64, 64])
    wrapper = DistillationWrapper(model, aligner)

    assert hasattr(wrapper, "stride"), "wrapper missing stride"
    assert hasattr(wrapper, "names"), "wrapper missing names"
    assert wrapper.stride is model.stride, "stride not delegated"
    assert wrapper.names is model.names, "names not delegated"
    print("TODO-5: DistillationWrapper attribute delegation ✓")


def test_wrapper_named_modules_includes_both():
    """wrapper.named_modules() includes student and aligner submodules."""
    from ultralytics.engine.distiller import DistillationWrapper
    from ultralytics.nn.modules.aligner import ConvAligner, MultiScaleAligner

    student = torch.nn.Linear(10, 10)
    aligner = MultiScaleAligner([64], [80])
    wrapper = DistillationWrapper(student, aligner)

    names = [name for name, _ in wrapper.named_modules()]
    assert any("student" in n for n in names), "student not in named_modules"
    assert any("aligner" in n for n in names), "aligner not in named_modules"
    print("TODO-5: DistillationWrapper named_modules includes both ✓")


# === TODO-6: distill_cfg loading ===

def test_distill_cfg_loading():
    """create_distiller loads distill_cfg YAML correctly."""
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    assert trainer.distill_cfg is not None, "distill_cfg should be loaded"
    assert "teacher" in trainer.distill_cfg, "missing teacher key"
    assert "student" in trainer.distill_cfg, "missing student key"
    print("TODO-6: distill_cfg loading ✓")


def test_distill_cfg_none_when_unset():
    """distill_cfg is None when not specified."""
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={"model": "yolov8n.yaml", "data": "coco8.yaml"})
    assert trainer.distill_cfg is None, "distill_cfg should be None"
    print("TODO-6: distill_cfg None when unset ✓")


# === TODO-7: Teacher loading & freeze ===

def test_teacher_freeze():
    """Teacher parameters are all frozen after _setup_teacher."""
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    trainer.device = torch.device("cpu")
    trainer._setup_teacher("yolov8s.pt")

    for p in trainer.teacher.parameters():
        assert not p.requires_grad, "Teacher param should be frozen"
    assert not trainer.teacher.training, "Teacher should be in eval mode"
    print("TODO-7: Teacher freeze ✓")


# === TODO-8: _resolve_module + hooks ===

def test_resolve_module_int():
    """Integer spec resolves to model.model[idx]."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    model = YOLO("yolov8n.pt").model
    module = trainer._resolve_module(model, 15)
    assert module is model.model[15], "Int spec should resolve to model.model[idx]"
    print("TODO-8: _resolve_module int ✓")


def test_resolve_module_str():
    """String spec resolves via get_submodule."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    model = YOLO("yolov8n.pt").model
    module = trainer._resolve_module(model, "model.22.cv2.0.1")
    expected = model.model[22].cv2[0][1]
    assert module is expected, "String spec should resolve to nested module"
    print("TODO-8: _resolve_module str ✓")


def test_feature_hooks_capture():
    """Forward hooks capture features from specified layers."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    model = YOLO("yolov8n.pt").model
    storage = []
    specs = ["model.22.cv2.0.1", "model.22.cv3.0.1"]
    handles = trainer._register_feature_hooks(model, specs, storage)

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        model(x)

    assert len(storage) == 2, f"Expected 2 features, got {len(storage)}"
    for h in handles:
        h.remove()
    print("TODO-8: Feature hooks capture ✓")


# === TODO-9: _get_layer_channels ===

def test_get_layer_channels():
    """Dummy forward correctly measures output channels."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
        "imgsz": 640,
    })
    trainer.device = torch.device("cpu")
    model = YOLO("yolov8n.pt").model
    specs = ["model.22.cv2.0.1", "model.22.cv3.0.1"]
    channels = trainer._get_layer_channels(model, specs)
    assert len(channels) == 2, f"Expected 2 channels, got {len(channels)}"
    assert all(c > 0 for c in channels), f"Channels should be positive, got {channels}"
    print(f"TODO-9: _get_layer_channels = {channels} ✓")


# === TODO-10: Aligner creation ===

def test_setup_aligner():
    """MultiScaleAligner created with correct number of aligners."""
    from ultralytics.engine.distiller import create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })
    trainer.device = torch.device("cpu")
    student_ch = [64, 64, 64, 80, 80, 80]
    teacher_ch = [128, 128, 128, 160, 160, 160]
    trainer._setup_aligner(student_ch, teacher_ch)
    assert len(trainer.aligner_module.aligners) == 6, "Should have 6 aligners"
    print("TODO-10: Aligner creation ✓")


# === TODO-11: build_optimizer includes aligner ===

def test_optimizer_includes_aligner():
    """build_optimizer on wrapper includes both student and aligner parameters."""
    from ultralytics import YOLO
    from ultralytics.engine.distiller import DistillationWrapper, create_distiller
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.modules.aligner import ConvAligner, MultiScaleAligner

    Distiller = create_distiller(DetectionTrainer)
    trainer = Distiller(overrides={
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "distill_cfg": str(DISTILL_CFG),
    })

    model = YOLO("yolov8n.pt").model
    aligner = MultiScaleAligner([64, 64], [64, 64])
    wrapper = DistillationWrapper(model, aligner)

    optimizer = trainer.build_optimizer(wrapper, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5)

    # Collect all optimizer params
    opt_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(id(p))

    # Check aligner params are included
    for p in aligner.parameters():
        assert id(p) in opt_params, "Aligner param missing from optimizer"
    print("TODO-11: build_optimizer includes aligner ✓")


if __name__ == "__main__":
    print("=== Phase 4 Tests ===\n")
    test_wrapper_attribute_delegation()
    test_wrapper_named_modules_includes_both()
    test_distill_cfg_loading()
    test_distill_cfg_none_when_unset()
    test_teacher_freeze()
    test_resolve_module_int()
    test_resolve_module_str()
    test_feature_hooks_capture()
    test_get_layer_channels()
    test_setup_aligner()
    test_optimizer_includes_aligner()
    print("\nAll Phase 4 tests passed!")
