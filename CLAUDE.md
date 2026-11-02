# Passk
Reliability and cost layer for Inspect AI eval logs. Python 3.10+, MIT.

## Commands
- Setup: `uv sync` (or `pip install -e .[dev]`)
- Test: `pytest -q`
- Lint and types: `ruff check . && pyright`

## Rules
- Zero spend: never add a paid service, paid API or paid runner.
- Secrets only from environment variables (`.env` is git-ignored); never log or print them.
- No live model calls in tests; use mocks or recorded cassettes.
- Wrap external formats (OpenTelemetry, OpenAI export) in one adapter module each.
- Every function with logic gets a unit test; estimators and parsers also get property tests.
- A README claim must link to the script and data that produce it.
- Heuristics are labeled heuristics; do not write "secure" or "guaranteed".

## Definition of done
Tests pass, types pass, docs updated, acceptance checks verified.
