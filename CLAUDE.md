# CLAUDE.md

## Language

Always respond in Korean.

## Build & Installation

```bash
pip install -e .                          # Editable install (dev)
pip install -e ".[dev,export]"            # With dev + export deps
pip install -e ".[dev,export,solutions,logging,extra,typing]"  # All optional deps
```

## Testing

```bash
pytest tests/                                          # All tests
pytest tests/test_python.py -v                         # Core functionality
pytest tests/test_python.py::test_train -v             # Single test
pytest tests/test_cli.py                               # CLI tests
pytest tests/ --slow                                   # Include slow tests
pytest tests/ --cov=ultralytics --cov-report=html      # With coverage
```

## Linting & Formatting

```bash
ruff format ultralytics/          # Format code
ruff check ultralytics/           # Lint
docformatter --check ultralytics/ # Docstring format
codespell ultralytics/            # Spell check
```

Line length: 120 characters. Google-style docstrings with type hints.

## CLI Usage

```bash
yolo TASK MODE ARGS
# Tasks: detect, segment, classify, pose, obb
# Modes: train, val, predict, export, track, benchmark
yolo detect train data=coco8.yaml model=yolo26n.pt epochs=10
yolo checks    # System info
```

## 소스 코드 아키텍처

`ultralytics/` 패키지에 전체 소스 코드가 위치합니다. 각 하위 디렉토리의 CLAUDE.md에서 영역별 상세 가이드를 확인하세요.

- `ultralytics/` — 소스 패키지 (@ultralytics/CLAUDE.md)
