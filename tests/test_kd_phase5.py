"""Phase 5 tests: engine __init__ exports."""


def test_import_create_distiller():
    from ultralytics.engine import create_distiller
    assert callable(create_distiller)


def test_import_distillation_wrapper():
    from ultralytics.engine import DistillationWrapper
    assert DistillationWrapper is not None


if __name__ == "__main__":
    test_import_create_distiller()
    test_import_distillation_wrapper()
    print("All Phase 5 tests passed!")
