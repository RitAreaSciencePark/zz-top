# Contributing to zztop

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/RitAreaSciencePark/zz-top.git
cd zztop
pip install -e ".[all]"
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Code style

- Follow PEP 8.
- Use type annotations for public functions.
- Add docstrings (NumPy style) to all public classes and functions.

## Pull requests

1. Fork the repository.
2. Create a feature branch from `main`.
3. Write tests for new functionality.
4. Make sure all tests pass.
5. Submit a pull request with a clear description of the changes.

## Reporting bugs

Open an issue with:
- A minimal reproducible example.
- Python version and OS.
- Output of `python -c "import zztop; print(zztop.__version__)"`.
