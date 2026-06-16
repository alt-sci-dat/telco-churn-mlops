# Repository Guidelines

## Project Structure & Module Organization
The package lives under `src/churn/` (src layout — tests import the *installed*
package, not loose files). Modules have single responsibilities and a strict
dependency direction: `config.py` (paths, feature lists, constants — the single
source of truth) is imported by everything; `data.py` (download + clean) and
`preprocess.py` (scikit-learn `ColumnTransformer`) are composed by `train.py`,
which produces one artifact: `models/churn_pipeline.joblib` (preprocessing +
XGBoost glued together). `drift.py` (PSI monitoring) and `schemas.py` (Pydantic
contracts) are shared by training and the API. `app/main.py` is the FastAPI
service; it loads the model via dependency injection (`get_model`,
`get_reference`) so tests can override it. Trained artifacts in `models/` are
committed so Docker/deploy need no training run.

## Build, Test, and Development Commands
- `make install` — create `.venv` and install dev + runtime deps (`pip install -e .`).
- `make train [TRIALS=50]` — run `python -m churn.train`; downloads data, tunes, logs to MLflow, writes `models/`.
- `make test` — run pytest. Single file: `pytest tests/test_api.py`; by keyword: `pytest -k drift`.
- `make lint` — `ruff check .`.
- `make serve` — `uvicorn app.main:app --reload`.
- `make mlflow` — MLflow UI on `sqlite:///mlflow.db`.
- `make docker-build` / `make docker-run` — build/run the container.

## Coding Style & Naming Conventions
Python, 4-space indent, 100-char lines. Lint/format with **ruff** (rules `E,F,I,UP,B`;
`E501` and `B008` ignored — `B008` because `Depends(...)` in defaults is the
intended FastAPI pattern). `snake_case` for functions/variables, `PascalCase` for
Pydantic models. Keep `data.clean_data` and `drift` functions pure (DataFrame in,
value out) so they unit-test without I/O. Any new feature column must be added to
the lists in `config.py` and the schema in `schemas.py` together.

## Testing Guidelines
pytest, configured in `pyproject.toml` (`testpaths = ["tests"]`). Tests use
synthetic fixtures from `tests/conftest.py` (`raw_df`, `training_frame`,
`fitted_pipeline`) — no dataset download or trained model required, so CI stays
fast. API tests use `TestClient` + `app.dependency_overrides`; clear
`_REQUEST_BUFFER` in fixtures to keep tests isolated. Add tests alongside the
module they cover (`test_<module>.py`).

## Commit & Pull Request Guidelines
No git history exists yet; use Conventional Commits (`feat:`, `fix:`, `test:`,
`docs:`, `ci:`). Keep subjects imperative and ≤72 chars. PRs should describe the
change, link issues, and must pass the CI workflow (`.github/workflows/ci.yml`:
ruff + pytest + Docker build) before merge.
