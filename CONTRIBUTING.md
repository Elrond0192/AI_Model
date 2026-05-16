# Contributing to Basketball Performance AI

Thank you for your interest in contributing! This guide explains the conventions,
development workflow, and quality standards for this project.

## Prerequisites

- Python 3.11 or 3.12
- `pip install -e ".[dev]"` to install in editable mode with dev dependencies
- (Optional) `pre-commit install` to activate pre-commit hooks

## Development workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes (see **Code conventions** below).
3. Run the test suite: `pytest tests/ -q --tb=short`
4. Run the linter: `ruff check basketball_ai/ --select E,F,W --ignore E501`
5. Open a pull request against `main`.

## Code conventions

- **Python style**: follow PEP 8; line length ≤ 120 chars; Google-style docstrings.
- **Type hints**: all public functions must have type annotations.
- **Tests**: add a test for every new feature or bug fix; target ≥ 80 % coverage in
  `basketball_ai/models/` and `basketball_ai/api/`.
- **Commits**: one logical change per commit; use the imperative mood
  ("Add feature X", not "Added feature X").
- **Logging**: use `logging.getLogger(__name__)`; never use `print()` in library code.
- **Security**: never commit secrets; use `.env` for local configuration.

## Running specific test suites

```bash
pytest tests/test_models.py   # Model tests
pytest tests/test_api.py      # API tests
pytest tests/test_chat.py     # Chat engine tests
pytest tests/ --cov=basketball_ai --cov-report=term-missing
```

## Adding a new chat intent

1. Add the intent to `basketball_ai/chat/intent.py` (`Intent` enum + examples).
2. Register a handler in `basketball_ai/chat/handlers.py` using `@register_handler(Intent.MY_INTENT)`.
3. Add tests in `tests/test_chat.py`.

## Architecture overview

See `docs/architecture/` for C4 diagrams. Key modules:

| Module | Responsibility |
|--------|---------------|
| `basketball_ai/data/` | Data loading, schema mapping, synthetic generator |
| `basketball_ai/features/` | Feature engineering (player, team, context) |
| `basketball_ai/models/` | XGBoost performance model, ensemble, age curve |
| `basketball_ai/scenarios/` | What-if engine, transfer analysis, lineup optimisation |
| `basketball_ai/chat/` | Intent detection, session management, response generation |
| `basketball_ai/api/` | FastAPI routes, auth middleware, rate limiting |
| `basketball_ai/auth/` | User management, RBAC, session tokens |
| `gui/` | Streamlit multi-user GUI |

## Reporting bugs

Open a GitHub Issue with: Python version, steps to reproduce, expected vs. actual behaviour,
and any relevant log output.
