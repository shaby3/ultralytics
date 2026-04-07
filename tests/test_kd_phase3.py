"""Phase 3 tests: distill_cfg in default config and example YAML."""

from pathlib import Path

from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils import YAML


def test_distill_cfg_in_defaults():
    assert "distill_cfg" in DEFAULT_CFG_DICT, "distill_cfg key missing from DEFAULT_CFG_DICT"


def test_distill_cfg_default_none():
    assert DEFAULT_CFG_DICT["distill_cfg"] is None, f"Expected None, got {DEFAULT_CFG_DICT['distill_cfg']}"


def test_distill_cfg_yaml_exists():
    path = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "distill_cfg.yaml"
    assert path.exists(), f"distill_cfg.yaml not found at {path}"


def test_distill_cfg_yaml_structure():
    path = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "distill_cfg.yaml"
    cfg = YAML.load(path)
    assert "teacher" in cfg, "Missing 'teacher' key"
    assert "student" in cfg, "Missing 'student' key"
    assert "model" in cfg["teacher"], "Missing 'teacher.model' key"
    assert "layers" in cfg["teacher"], "Missing 'teacher.layers' key"
    assert "layers" in cfg["student"], "Missing 'student.layers' key"
    assert len(cfg["teacher"]["layers"]) == len(cfg["student"]["layers"]), "teacher/student layer count mismatch"
    assert len(cfg["teacher"]["layers"]) == 6, f"Expected 6 layers, got {len(cfg['teacher']['layers'])}"


if __name__ == "__main__":
    test_distill_cfg_in_defaults()
    test_distill_cfg_default_none()
    test_distill_cfg_yaml_exists()
    test_distill_cfg_yaml_structure()
    print("All Phase 3 tests passed!")
