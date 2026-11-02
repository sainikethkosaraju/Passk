# Contributing
`pip install -e .[dev]`, then `pytest && ruff check . && pyright` must pass. One change per PR.
Estimator and parser changes need a property test. No live model calls in tests.
