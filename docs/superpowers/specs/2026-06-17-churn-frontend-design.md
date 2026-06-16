# Churn Predictor Web UI — Design

**Date:** 2026-06-17
**Status:** Approved (design), pending implementation plan

## Goal

Add a simple, polished web UI so a non-technical viewer (e.g. a recruiter) can
fill in a customer's details, get a churn prediction with a clear risk
visualization, and see a live data-drift status — turning the API from "plumbing"
into a usable product demo. Must remain free to host and deploy with the existing
Docker image / Hugging Face Space.

## Non-goals (YAGNI)

- No authentication, no database, no user accounts.
- No JavaScript build step, no SPA framework, no charting library.
- No batch-prediction UI.
- No new runtime dependencies (keep the image slim).

## Architecture

- A single self-contained `app/static/index.html` — HTML + CSS + vanilla JS, plus
  inline SVG for the gauge and CSS bars for drift. No assets, no bundler.
- The existing FastAPI app serves it at `GET /` via `FileResponse`. All API calls
  are same-origin to existing endpoints, so there is no CORS configuration.
- Ships inside the current Docker image (already `COPY`s `app/`) and the same HF
  Space. No infrastructure changes.

### Backend changes (`app/main.py`)

1. `GET /` — return `app/static/index.html` as `FileResponse` (replaces the
   current JSON greeting). The JSON greeting is removed.
2. `GET /metadata` — return the contents of `models/model_metadata.json`
   (`metrics`, `best_params`, `churn_rate`, `n_train`, `n_test`). If the file is
   missing, return `503`. This lets the UI display the model's real metrics rather
   than hardcoded values.
3. Imports: `FileResponse` from `fastapi.responses`; `Path` to locate the static
   file. No new third-party dependencies.

No changes to `/predict`, `/predict/batch`, `/drift`, `/health`.

## UI layout (light SaaS, card-based, responsive)

- **Header:** title "Telco Churn Predictor", subtitle, and a badge showing
  `XGBoost · ROC-AUC <value from /metadata>`.
- **Left card — "Customer details":** the 19 model inputs grouped into
  *Account / Services / Billing*. Select/number inputs with valid options matching
  the Pydantic schema. Sensible defaults pre-filled. Buttons: **Load example**
  (fills a known high-risk customer) and **Predict**.
- **Right card — "Prediction":** an SVG arc gauge showing churn probability
  (0–100%), a colored risk pill (low / medium / high), and a plain-English verdict
  ("Likely to churn" / "Likely to stay"). Empty state shown before first run.
- **Bottom card — "Data drift monitor":** a **Run drift check** button that calls
  `/drift`; renders each feature's PSI as a horizontal bar colored by severity
  (none = green, moderate = amber, significant = red) plus an overall status line.
  Short explainer: drift reflects the customers scored so far this session.

### Visual style

Light theme, generous whitespace, one accent color, card surfaces with subtle
shadows, system font stack. Mobile: cards stack vertically.

## Data flow

1. On load: `GET /metadata` and `GET /health` → populate header badge + a small
   "model loaded" indicator.
2. Submit form: serialize inputs to JSON (numbers cast to numbers) → `POST
   /predict` → render gauge + pill + verdict.
3. Run drift check: `GET /drift` → render PSI bars + status.

## Error handling

- `POST /predict` `422`: show a clear inline message (form constrains inputs, so
  this is a safety net).
- `GET /drift` empty buffer or `503`: show "Make a few predictions first, then
  check drift." (not an error state).
- Any network/fetch failure: inline error banner; never fail silently.

## Testing

- `tests/test_api.py`:
  - `GET /` returns `200` with `content-type: text/html`.
  - `GET /metadata` returns `200` with a JSON body containing a `metrics` object
    (uses the committed metadata file present in the repo).
- JavaScript is intentionally minimal and untested by an automated harness (out of
  scope); verified manually in the browser locally and on the live Space after
  deploy.

## Deployment

Unchanged. The same `docker build` includes `app/static/index.html`; the same HF
Space serves the UI at its root URL. After merge + push, verify the live Space
root renders and `/predict` works from the form.

## Acceptance criteria

- Visiting `/` renders the UI; "Load example" + "Predict" returns a probability
  and risk level consistent with the API.
- The header shows the real ROC-AUC from `/metadata`.
- "Run drift check" shows per-feature PSI after some predictions, and a friendly
  message before any.
- All existing tests still pass, plus the two new endpoint tests.
- Local Docker image serves the UI; live HF Space serves the UI after deploy.
