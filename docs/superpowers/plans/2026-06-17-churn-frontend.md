# Churn Predictor Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single FastAPI-served web page that lets a user enter a customer, get a churn prediction with a risk gauge, and view a live data-drift report.

**Architecture:** One self-contained `app/static/index.html` (HTML + CSS + vanilla JS, inline SVG) served at `GET /` via `FileResponse`. A new `GET /metadata` endpoint exposes the trained model's real metrics. All calls are same-origin to existing endpoints — no CORS, no new dependencies, ships in the existing Docker image / HF Space.

**Tech Stack:** FastAPI (FileResponse), vanilla JavaScript, SVG/CSS. No build step, no JS framework, no charting library.

---

## File Structure

- **Modify** `app/main.py` — add `GET /metadata`; change `GET /` to serve the HTML page.
- **Create** `app/static/index.html` — the entire UI (markup + styles + script).
- **Modify** `tests/test_api.py` — add tests for `/` (HTML) and `/metadata`.

The Dockerfile already `COPY app/ ./app/`, so `app/static/` is included with no Dockerfile change.

---

### Task 1: `GET /metadata` endpoint

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
def test_metadata_returns_metrics(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    assert "roc_auc" in body["metrics"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_metadata_returns_metrics -v`
Expected: FAIL — route does not exist yet, returns `404` (assert `200` fails).

- [ ] **Step 3: Add imports and the endpoint**

In `app/main.py`, add to the imports near the top (after `from functools import lru_cache`):

```python
import json
from pathlib import Path
```

And add `FileResponse` to the FastAPI responses import (create this line with the other imports):

```python
from fastapi.responses import FileResponse
```

Then add the endpoint (place it just above the existing `root()` function):

```python
@app.get("/metadata", tags=["ops"])
def metadata() -> dict:
    """Return the trained model's metrics so the UI can display real numbers."""
    if not config.METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    return json.loads(config.METADATA_PATH.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_metadata_returns_metrics -v`
Expected: PASS (the committed `models/model_metadata.json` is read).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add GET /metadata endpoint for the UI"
```

---

### Task 2: Serve the UI page at `GET /`

**Files:**
- Create: `app/static/index.html`
- Modify: `app/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
def test_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Telco Churn Predictor" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_root_serves_html -v`
Expected: FAIL — `/` currently returns JSON (`content-type: application/json`).

- [ ] **Step 3: Create the UI file**

Create `app/static/index.html` with this exact content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Telco Churn Predictor</title>
<style>
  :root {
    --accent: #4f46e5; --accent-soft: #eef2ff; --bg: #f7f8fb; --card: #ffffff;
    --text: #1f2937; --muted: #6b7280; --border: #e5e7eb;
    --green: #16a34a; --amber: #d97706; --red: #dc2626;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 20px 24px; }
  header h1 { margin: 0; font-size: 20px; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
  .badge { display: inline-block; margin-top: 8px; background: var(--accent-soft);
    color: var(--accent); font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; }
  main { max-width: 1080px; margin: 24px auto; padding: 0 16px; display: grid;
    grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .card h2 { margin: 0 0 14px; font-size: 16px; }
  .full { grid-column: 1 / -1; }
  .group-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); margin: 14px 0 8px; }
  .field { display: flex; flex-direction: column; margin-bottom: 10px; }
  .field label { font-size: 13px; margin-bottom: 4px; color: var(--text); }
  select, input { padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px;
    font-size: 14px; background: #fff; color: var(--text); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .btns { display: flex; gap: 10px; margin-top: 14px; }
  button { font-size: 14px; font-weight: 600; border-radius: 8px; padding: 10px 16px; cursor: pointer; border: 1px solid var(--border); background: #fff; color: var(--text); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .6; cursor: default; }
  .gauge-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 240px; text-align: center; }
  .gauge { position: relative; width: 200px; height: 200px; }
  .gauge .pct { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-size: 38px; font-weight: 700; }
  .pill { display: inline-block; margin-top: 12px; font-weight: 700; font-size: 13px; padding: 6px 14px; border-radius: 999px; }
  .verdict { margin-top: 10px; color: var(--muted); font-size: 14px; }
  .empty { color: var(--muted); font-size: 14px; }
  .drift-row { display: grid; grid-template-columns: 160px 1fr 70px; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 13px; }
  .track { background: #f1f3f7; border-radius: 6px; height: 12px; overflow: hidden; }
  .fill { height: 100%; border-radius: 6px; }
  .status { padding: 10px 12px; border-radius: 8px; font-size: 14px; margin-bottom: 12px; font-weight: 600; }
  .banner { display: none; background: #fef2f2; color: var(--red); border: 1px solid #fecaca;
    padding: 10px 12px; border-radius: 8px; font-size: 14px; margin-bottom: 12px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
  @media (max-width: 760px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>📉 Telco Churn Predictor</h1>
  <p>Predict whether a customer will cancel their service, with live data-drift monitoring.</p>
  <span class="badge" id="badge">Loading model…</span>
</header>

<main>
  <section class="card">
    <h2>Customer details</h2>
    <div id="formBanner" class="banner"></div>
    <form id="form">
      <div class="group-label">Account</div>
      <div class="row">
        <div class="field"><label>Gender</label>
          <select name="gender"><option>Female</option><option>Male</option></select></div>
        <div class="field"><label>Senior citizen</label>
          <select name="SeniorCitizen"><option value="0">No</option><option value="1">Yes</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Partner</label>
          <select name="Partner"><option>No</option><option>Yes</option></select></div>
        <div class="field"><label>Dependents</label>
          <select name="Dependents"><option>No</option><option>Yes</option></select></div>
      </div>
      <div class="field"><label>Tenure (months)</label>
        <input type="number" name="tenure" min="0" max="120" value="5" /></div>

      <div class="group-label">Services</div>
      <div class="row">
        <div class="field"><label>Phone service</label>
          <select name="PhoneService"><option>Yes</option><option>No</option></select></div>
        <div class="field"><label>Multiple lines</label>
          <select name="MultipleLines"><option>No</option><option>Yes</option><option>No phone service</option></select></div>
      </div>
      <div class="field"><label>Internet service</label>
        <select name="InternetService"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
      <div class="row">
        <div class="field"><label>Online security</label>
          <select name="OnlineSecurity"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
        <div class="field"><label>Online backup</label>
          <select name="OnlineBackup"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Device protection</label>
          <select name="DeviceProtection"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
        <div class="field"><label>Tech support</label>
          <select name="TechSupport"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Streaming TV</label>
          <select name="StreamingTV"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
        <div class="field"><label>Streaming movies</label>
          <select name="StreamingMovies"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
      </div>

      <div class="group-label">Billing</div>
      <div class="field"><label>Contract</label>
        <select name="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
      <div class="field"><label>Paperless billing</label>
        <select name="PaperlessBilling"><option>Yes</option><option>No</option></select></div>
      <div class="field"><label>Payment method</label>
        <select name="PaymentMethod"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></div>
      <div class="row">
        <div class="field"><label>Monthly charges ($)</label>
          <input type="number" name="MonthlyCharges" min="0" max="1000" step="0.05" value="89.10" /></div>
        <div class="field"><label>Total charges ($)</label>
          <input type="number" name="TotalCharges" min="0" max="100000" step="0.05" value="445.50" /></div>
      </div>

      <div class="btns">
        <button type="submit" class="primary" id="predictBtn">Predict churn</button>
        <button type="button" id="exampleBtn">Load example</button>
      </div>
    </form>
  </section>

  <section class="card">
    <h2>Prediction</h2>
    <div class="gauge-wrap" id="result">
      <p class="empty">Fill in the form and click <strong>Predict churn</strong> to see the result.</p>
    </div>
  </section>

  <section class="card full">
    <h2>Data drift monitor</h2>
    <div id="driftBanner" class="banner"></div>
    <div id="driftBody"><p class="empty">Make a few predictions, then run a drift check.</p></div>
    <div class="btns"><button type="button" id="driftBtn">Run drift check</button></div>
    <p class="hint">Drift compares the customers you've scored this session against the training data (PSI: &lt;0.1 none, 0.1–0.25 moderate, &gt;0.25 significant).</p>
  </section>
</main>

<script>
const NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"];
const HIGH_RISK_EXAMPLE = {
  gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No", tenure: "1",
  PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
  OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
  StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month",
  PaperlessBilling: "Yes", PaymentMethod: "Electronic check",
  MonthlyCharges: "95.00", TotalCharges: "95.00"
};

function riskColor(level) {
  return level === "high" ? "var(--red)" : level === "medium" ? "var(--amber)" : "var(--green)";
}
function showBanner(id, msg) { const b = document.getElementById(id); b.textContent = msg; b.style.display = msg ? "block" : "none"; }

async function loadMetadata() {
  try {
    const r = await fetch("/metadata");
    if (!r.ok) throw new Error();
    const m = await r.json();
    const auc = m.metrics && m.metrics.roc_auc != null ? Number(m.metrics.roc_auc).toFixed(3) : "n/a";
    document.getElementById("badge").textContent = "XGBoost · ROC-AUC " + auc;
  } catch { document.getElementById("badge").textContent = "XGBoost model"; }
}

function gaugeHTML(prob, level) {
  const pct = Math.round(prob * 100);
  const r = 80, c = 2 * Math.PI * r, off = c * (1 - prob), col = riskColor(level);
  return `
    <div class="gauge">
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="${r}" fill="none" stroke="#eef0f5" stroke-width="16"/>
        <circle cx="100" cy="100" r="${r}" fill="none" stroke="${col}" stroke-width="16"
          stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}"
          transform="rotate(-90 100 100)"/>
      </svg>
      <div class="pct">${pct}%</div>
    </div>
    <span class="pill" style="background:${col}1a;color:${col}">${level.toUpperCase()} RISK</span>
    <div class="verdict">${prob >= 0.5 ? "Likely to churn" : "Likely to stay"} — ${pct}% churn probability</div>`;
}

function buildPayload() {
  const data = {};
  document.querySelectorAll("#form [name]").forEach(el => {
    data[el.name] = NUMERIC.includes(el.name) ? Number(el.value) : el.value;
  });
  return data;
}

document.getElementById("exampleBtn").addEventListener("click", () => {
  for (const [k, v] of Object.entries(HIGH_RISK_EXAMPLE)) {
    const el = document.querySelector(`#form [name="${k}"]`);
    if (el) el.value = v;
  }
});

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showBanner("formBanner", "");
  const btn = document.getElementById("predictBtn");
  btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const r = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload())
    });
    if (r.status === 422) { showBanner("formBanner", "Some inputs are invalid. Please review and try again."); return; }
    if (!r.ok) throw new Error("Server error " + r.status);
    const res = await r.json();
    document.getElementById("result").innerHTML = gaugeHTML(res.churn_probability, res.risk_level);
  } catch (err) {
    showBanner("formBanner", "Could not reach the prediction service. " + err.message);
  } finally { btn.disabled = false; btn.textContent = "Predict churn"; }
});

document.getElementById("driftBtn").addEventListener("click", async () => {
  showBanner("driftBanner", "");
  const body = document.getElementById("driftBody");
  try {
    const r = await fetch("/drift");
    if (r.status === 503) { showBanner("driftBanner", "Drift reference not available."); return; }
    if (!r.ok) throw new Error("Server error " + r.status);
    const d = await r.json();
    if (!d.n_samples) { body.innerHTML = `<p class="empty">${d.message || "Make a few predictions first."}</p>`; return; }
    const col = d.drift_detected ? "var(--red)" : "var(--green)";
    let html = `<div class="status" style="background:${col}1a;color:${col}">${d.message} (${d.n_samples} samples)</div>`;
    for (const [feat, info] of Object.entries(d.features)) {
      const lvl = info.drift, fc = lvl === "significant" ? "var(--red)" : lvl === "moderate" ? "var(--amber)" : "var(--green)";
      const w = Math.min(info.psi / 0.5, 1) * 100;
      html += `<div class="drift-row"><span>${feat}</span>
        <span class="track"><span class="fill" style="width:${w}%;background:${fc}"></span></span>
        <span>${info.psi.toFixed(3)}</span></div>`;
    }
    body.innerHTML = html;
  } catch (err) { showBanner("driftBanner", "Could not load drift report. " + err.message); }
});

loadMetadata();
</script>
</body>
</html>
```

- [ ] **Step 4: Change `GET /` to serve the page**

In `app/main.py`, add a module-level constant after the `_REQUEST_BUFFER` definition:

```python
STATIC_DIR = Path(__file__).resolve().parent / "static"
```

Then replace the existing `root()` function:

```python
@app.get("/", tags=["ops"])
def root() -> dict:
    return {"message": "Telco Churn API. See /docs for interactive documentation."}
```

with:

```python
@app.get("/", response_class=FileResponse, tags=["ops"])
def root() -> FileResponse:
    """Serve the single-page web UI."""
    return FileResponse(STATIC_DIR / "index.html")
```

- [ ] **Step 5: Run the new test and the full suite**

Run: `pytest tests/test_api.py::test_root_serves_html -v`
Expected: PASS.

Run: `pytest -q`
Expected: PASS (all previous tests + the two new ones).

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html app/main.py tests/test_api.py
git commit -m "feat: add single-page web UI served at /"
```

---

### Task 3: Local browser verification

**Files:** none (manual verification).

- [ ] **Step 1: Start the server**

Run (in the activated venv): `uvicorn app.main:app --port 8000`

- [ ] **Step 2: Verify the page and a prediction**

In a browser, open `http://localhost:8000/`. Confirm:
- The header badge shows `XGBoost · ROC-AUC 0.847` (loaded from `/metadata`).
- Click **Load example**, then **Predict churn** → the gauge shows ~92% and a red **HIGH RISK** pill.
- Change Contract to "Two year" and tenure to 70 → predict again → low probability, green pill.
- Click **Run drift check** → after the predictions above, per-feature PSI bars render with an overall status line.

Command-line smoke check (optional, while server runs):
`curl -s localhost:8000/ | grep -c "Telco Churn Predictor"` → expect `1`.

- [ ] **Step 3: Stop the server** (Ctrl-C).

---

### Task 4: Docker rebuild + container verification

**Files:** none (the Dockerfile already copies `app/`).

- [ ] **Step 1: Rebuild the image**

Run: `docker build -t telco-churn-api .`
Expected: build succeeds (exit 0).

- [ ] **Step 2: Run and verify the UI is served from the container**

```bash
docker rm -f churn_ui >/dev/null 2>&1
docker run -d --name churn_ui -p 8000:8000 telco-churn-api
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/        # expect 200
curl -s localhost:8000/ | grep -c "Telco Churn Predictor"             # expect 1
curl -s localhost:8000/metadata | grep -c roc_auc                      # expect 1
docker rm -f churn_ui
```

Expected: `200`, `1`, `1`.

---

### Task 5: Deploy

**Files:** none.

- [ ] **Step 1: Push to GitHub (triggers CI)**

```bash
git push origin main
```

- [ ] **Step 2: Push to the Hugging Face Space** (requires the user's HF write token)

```bash
git push space main:main
```
When prompted: username `nishusinghrajput`, password = HF write token.

- [ ] **Step 3: Verify the live Space**

After the HF build finishes (watch the Space "Logs" → "Build"), confirm:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://nishusinghrajput-telco-churn-mlops.hf.space/   # expect 200
```
Then open `https://nishusinghrajput-telco-churn-mlops.hf.space/` in a browser and run a prediction from the form.

---

## Self-Review

**Spec coverage:**
- Single FastAPI-served HTML page → Task 2. ✓
- `GET /metadata` endpoint → Task 1. ✓
- Three cards (form / prediction gauge / drift monitor) → index.html in Task 2. ✓
- Light SaaS style, responsive → CSS in Task 2. ✓
- Error handling (422, drift empty/503, network) → JS handlers + banners in Task 2. ✓
- Tests for `/` (HTML) and `/metadata` → Tasks 1 & 2. ✓
- Deploy via existing image / HF Space → Tasks 4 & 5. ✓
- No new dependencies → only `FileResponse` (built into FastAPI). ✓

**Placeholder scan:** No TBD/TODO; full HTML and code provided. ✓

**Type/name consistency:** Field `name` attributes match the `CustomerFeatures` schema; `NUMERIC` list matches `config.NUMERIC_FEATURES` (tenure, MonthlyCharges, TotalCharges, SeniorCitizen); `/predict` response keys (`churn_probability`, `risk_level`) and `/drift` keys (`n_samples`, `drift_detected`, `message`, `features[*].psi`, `features[*].drift`) match `schemas.py` and `drift.compute_drift`. ✓
