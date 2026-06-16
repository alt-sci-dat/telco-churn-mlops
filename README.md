# Telco Customer Churn — Production-Grade MLOps Project

Predict which telecom customers are about to cancel ("churn"), served as a real
web API with experiment tracking, automated testing, containerization, CI/CD, and
data-drift monitoring — built entirely with **free tools**.

This README doubles as a **from-zero tutorial**. Every concept is explained in
plain English before you use it. If you've never done ML or "MLOps" before, read
top to bottom.

---

## Table of contents

1. [What is this and why does it matter?](#1-what-is-this-and-why-does-it-matter)
2. [The big picture: what is MLOps?](#2-the-big-picture-what-is-mlops)
3. [Quickstart (run it in 5 minutes)](#3-quickstart)
4. [The dataset](#4-the-dataset)
5. [Part 1 — Data preprocessing pipeline](#5-part-1--data-preprocessing-pipeline)
6. [Part 2 — XGBoost model + Optuna tuning](#6-part-2--xgboost--optuna)
7. [Part 3 — MLflow experiment tracking](#7-part-3--mlflow-experiment-tracking)
8. [Part 4 — FastAPI prediction service](#8-part-4--fastapi-service)
9. [Part 5 — Docker containerization](#9-part-5--docker)
10. [Part 6 — GitHub Actions CI/CD](#10-part-6--cicd)
11. [Part 7 — Data drift monitoring](#11-part-7--data-drift-monitoring)
12. [Part 8 — Testing with pytest](#12-part-8--testing)
13. [Deploy for free](#13-deploy-for-free)
14. [The free-tools cheat sheet](#14-free-tools-cheat-sheet)
15. [Resume talking points](#15-resume-talking-points)

---

## 1. What is this and why does it matter?

A model that only runs in a Jupyter notebook on your laptop is a science project.
A model that other software can call over the internet, that's tested
automatically, that you can rebuild identically anywhere, and that tells you when
it's going stale — that's a **product**. This repo is the second kind. That gap
is exactly what "production-grade" and "MLOps" mean, and it's what makes a resume
project stand out.

**What the project does:** given a customer's details (how long they've been a
customer, their contract type, monthly charges, etc.), it returns the probability
they'll cancel their service, so a business could intervene (offer a discount,
call them) *before* they leave.

---

## 2. The big picture: what is MLOps?

**ML** = Machine Learning: writing code that learns patterns from data instead of
being explicitly programmed with rules.

**Ops** = Operations: the discipline of running software reliably in production
(deployment, monitoring, automation).

**MLOps** = applying software-operations rigor to machine learning. A normal app
has code + infrastructure. An ML app also has **data** and a **trained model**,
which both change over time and can rot silently. MLOps is the set of practices
that keep an ML system trustworthy: version your data and models, track every
experiment, test automatically, deploy reproducibly, and monitor for the data
changing underneath you.

Here's how the 8 pieces of this project fit together:

```
                 ┌──────────────────┐
   raw data ───▶ │ 1. Preprocessing │ ──┐
                 └──────────────────┘   │
                                        ▼
                 ┌──────────────────────────────┐     ┌────────────────────┐
                 │ 2. XGBoost + Optuna tuning    │ ──▶ │ 3. MLflow tracking │
                 └──────────────────────────────┘     │  (the lab notebook) │
                                │                      └────────────────────┘
                                ▼ saves one artifact
                   models/churn_pipeline.joblib
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 4. FastAPI service /predict   │ ◀── validates every request
                 │    + 7. drift monitoring      │
                 └──────────────────────────────┘
                                │ packaged by
                                ▼
                 ┌──────────────┐    ┌───────────────────────┐    ┌──────────┐
                 │ 5. Docker    │ ──▶│ 6. GitHub Actions CI  │ ──▶│ 13. Deploy│
                 │  (the box)   │    │  (8. tests on push)   │    │  (free)  │
                 └──────────────┘    └───────────────────────┘    └──────────┘
```

---

## 3. Quickstart

```bash
# 1. Install everything into an isolated virtual environment
make install
source .venv/bin/activate     # activate it (Windows: .venv\Scripts\activate)

# 2. Train the model (downloads the dataset automatically, ~1-2 min)
make train                     # or: make train TRIALS=50  (more tuning, slower)

# 3. Run the tests
make test

# 4. Start the API
make serve
# open http://localhost:8000/docs  -> click "Try it out" on /predict
```

Don't have `make`? Every command has a plain equivalent — see the [Makefile](Makefile)
or the per-part sections below.

---

## 4. The dataset

The **Telco Customer Churn** dataset (originally from IBM, mirrored on Kaggle):
~7,000 customers, 19 features, and a `Churn` Yes/No label. About **26.5%** of
customers churned — an *imbalanced* dataset, which we handle in Part 2.

You don't need a Kaggle account: `src/churn/data.py` downloads it from a public
mirror automatically and caches it in `data/raw/`. (If you *do* want Kaggle: the
free tier just requires a free account → Account → "Create New API Token" → save
the file to `~/.kaggle/kaggle.json`. The code will use it as a fallback.)

---

## 5. Part 1 — Data preprocessing pipeline

**Concept.** Raw data is messy. Models need clean, numeric input. A *preprocessing
pipeline* is the sequence of cleaning + transformation steps. The key MLOps idea:
**the exact same steps must run at training time and at prediction time.** If you
clean training data one way and live data another, your model silently breaks.
scikit-learn `Pipeline` objects guarantee they stay in sync.

**What we do** (`src/churn/data.py` + `src/churn/preprocess.py`):
- Fix the dataset's known quirks: `TotalCharges` is stored as text with 11 blank
  values → convert to numbers, impute the blanks. Map `Churn` Yes/No → 1/0.
- Numeric columns → fill missing values with the median.
- Categorical columns (e.g. `Contract`) → **one-hot encode** (turn each category
  into its own 0/1 column). `handle_unknown="ignore"` means an unseen category in
  a live request won't crash the API.
- We deliberately **don't scale** numbers: XGBoost is a tree model and splits on
  thresholds, so feature scale is irrelevant.

```bash
python -c "from churn.data import load_dataset; print(load_dataset().shape)"
```

---

## 6. Part 2 — XGBoost + Optuna

**XGBoost** (`src/churn/train.py`). A "gradient boosted trees" model: it builds
many small decision trees in sequence, each correcting the previous ones' errors.
It's the default winner for *tabular* data (spreadsheet-like rows and columns).

**Class imbalance.** Only 26.5% of customers churn. A lazy model could predict
"nobody churns" and be 73% accurate while being useless. We fix this two ways:
(1) `scale_pos_weight` tells XGBoost to weight the rare "churn" class more, and
(2) we optimize **ROC-AUC** (how well the model *ranks* churners above
non-churners), not raw accuracy.

**Optuna** = automated hyperparameter tuning. Hyperparameters are the model's
"knobs" you set before training (tree depth, learning rate, …). Hand-tuning is
slow guesswork. Optuna runs many "trials", intelligently searching for the best
combination. Each trial is scored with **5-fold cross-validation** (train on 4/5
of the data, test on the held-out 1/5, five times, average the scores) so we
don't overfit to one lucky split.

```bash
python -m churn.train --n-trials 30
```

Our trained model scores **ROC-AUC ≈ 0.85** on the held-out test set — solid for
this dataset.

---

## 7. Part 3 — MLflow experiment tracking

**Concept.** Every training run produces parameters, metrics, and a model file.
Without tracking, "which settings gave me that good result last Tuesday?" is
unanswerable. **Experiment tracking** is a lab notebook that auto-records every
run so you can compare and reproduce them.

**What we use.** MLflow is free and open-source. We run it locally with a SQLite
backend (no server, no account, no cost). `train.py` logs the best
hyperparameters, all metrics, and the model itself on every run.

```bash
make mlflow      # then open http://localhost:5000
```

> Want free *hosted* tracking instead of local? **DagsHub** gives every repo a
> free hosted MLflow server — just point `MLFLOW_TRACKING_URI` at it.

---

## 8. Part 4 — FastAPI service

**Concept.** An **API** lets other programs ask your model questions over HTTP.
**FastAPI** is a Python web framework that (a) validates input automatically from
your schema and (b) generates interactive docs for free.

**Input validation** (`src/churn/schemas.py`). Before the model sees anything, we
check the request has the right fields, types, and ranges. Send `tenure: -5` or
`Contract: "Forever"` and you get a clear **422** error — garbage never reaches
the model.

**Endpoints** (`app/main.py`):

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | liveness probe (deploy platforms ping this) |
| POST | `/predict` | score one customer |
| POST | `/predict/batch` | score many at once |
| GET  | `/drift` | drift report on recent traffic (Part 7) |
| GET  | `/docs` | interactive Swagger UI |

```bash
make serve
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.0,"TotalCharges":95.0}'
# -> {"churn_probability":0.92,"churn_prediction":1,"risk_level":"high"}
```

The model is loaded once at startup via FastAPI **dependency injection**, which
also lets tests swap in a fake model — no real training needed in CI.

---

## 9. Part 5 — Docker

**Concept.** "It works on my machine" is the oldest problem in software. **Docker**
packs your code + Python + every dependency at exact versions into an *image* — a
self-contained box. Any machine with Docker runs the identical box, so it behaves
the same on your laptop, in CI, and on the deploy host. A running box is a
*container*.

```bash
make docker-build
make docker-run        # API now on http://localhost:8000
```

The [`Dockerfile`](Dockerfile) installs dependencies in a cached layer (fast
rebuilds), copies the code + the trained model, runs as a non-root user
(security), and listens on `$PORT` (which Render/HF set automatically).

---

## 10. Part 6 — CI/CD

**Concept.** **CI** (Continuous Integration): every time you push code, a fresh
cloud machine automatically installs the project and runs the linter + tests, so
bugs are caught in minutes on a clean machine — not in production. **CD**
(Continuous Delivery): extend that to auto-build/deploy when tests pass.

**What we use.** GitHub Actions — **free** for public repos (and generous free
minutes for private ones). The workflow is in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). On every push/PR to
`main` it runs two jobs in parallel:
1. **test** — lint with ruff, then run pytest.
2. **docker-build** — prove the Docker image still builds.

The tests use synthetic data, so CI needs no dataset download and no GPU — it
runs in well under a minute.

---

## 11. Part 7 — Data drift monitoring

**Concept.** Your model learned from training data. If live traffic starts to
look *different* (a new marketing campaign brings in totally different customers),
predictions degrade — with **no error and no code change**. This is **data
drift**, and monitoring for it is your smoke detector.

**What we use** (`src/churn/drift.py`). The **Population Stability Index (PSI)**,
an industry standard (common in credit scoring). It bins each feature and
compares the training distribution ("expected") to live traffic ("actual").

| PSI | Meaning |
|-----|---------|
| < 0.10 | no significant drift |
| 0.10–0.25 | moderate drift, watch it |
| > 0.25 | significant drift, consider retraining |

At training time we save a small `drift_reference.json` (the training
distribution). The API buffers recent requests; hit `GET /drift` to compare live
traffic against that baseline.

```bash
# after sending some /predict requests:
curl http://localhost:8000/drift
```

---

## 12. Part 8 — Testing

**Concept.** Tests are code that checks your code. They let you change things
fearlessly: if you break something, a test fails immediately instead of a user
finding out. **pytest** is the standard Python testing tool.

**What we test** (`tests/`):
- `test_data.py` — the data-cleaning quirks (blank `TotalCharges`, target mapping).
- `test_preprocess.py` — the pipeline transforms correctly and survives unseen
  categories and missing values.
- `test_drift.py` — PSI is ~0 for identical data and high for shifted data.
- `test_api.py` — endpoints work, and bad input is rejected with 422.

All tests use small **synthetic** data and a tiny in-memory model, so the suite
runs in under a second and needs no dataset or trained model.

```bash
make test                       # all tests
pytest tests/test_api.py        # one file
pytest -k drift                 # tests matching "drift"
```

---

## 13. Deploy for free

You committed the trained model (`models/`, ~340 KB), so deployment needs no
training step. Two free options — pick one.

### Option A — Render (free web service)

Render's **free tier** runs a web service that sleeps after inactivity and wakes
on the next request (fine for a demo/portfolio).

1. Push this repo to GitHub (see below).
2. Go to [render.com](https://render.com) → sign up free → **New → Web Service** →
   connect your GitHub repo.
3. Render auto-detects the `Dockerfile`. Set:
   - **Environment:** Docker
   - **Instance type:** Free
   - (No build/start command needed — the Dockerfile handles it.)
4. Click **Create Web Service**. In a few minutes you get a public URL like
   `https://telco-churn.onrender.com`. Visit `/docs` to use it.

Render injects `$PORT` automatically, which our Dockerfile already respects.

### Option B — Hugging Face Spaces (free, Docker SDK)

HF **Spaces** host apps for free.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. **New → Space** → choose **Docker** as the SDK → blank template.
3. Add this line to the top of the Space's `README.md` so HF uses the right port:
   ```yaml
   ---
   title: Telco Churn API
   sdk: docker
   app_port: 8000
   ---
   ```
4. Push this repo's files to the Space's git remote. HF builds the Dockerfile and
   serves it at `https://<user>-<space>.hf.space`. Hit `/docs`.

> Either way, keep your trained `models/` committed. In a larger system you'd
> instead pull the model from a registry (MLflow/S3) at startup, but committing a
> small model is the pragmatic free-tier choice.

### Push to GitHub first

```bash
git init && git add -A && git commit -m "Initial commit: Telco churn MLOps project"
gh repo create telco-churn-mlops --public --source=. --push   # uses the GitHub CLI
```

The moment you push, the CI pipeline (Part 6) runs automatically — check the
**Actions** tab.

---

## 14. Free-tools cheat sheet

| Need | Tool | Free tier |
|------|------|-----------|
| Language / libs | Python, pandas, scikit-learn, XGBoost | open source |
| Tuning | Optuna | open source |
| Experiment tracking | MLflow (local SQLite) | open source; DagsHub for free hosted |
| API framework | FastAPI + Uvicorn | open source |
| Containers | Docker | free (Docker Desktop / Engine) |
| CI/CD | GitHub Actions | free for public repos |
| Testing / lint | pytest, ruff | open source |
| Hosting | Render **or** Hugging Face Spaces | free tier |
| Dataset | Telco Churn (public mirror / Kaggle) | free |

**Total cost: $0.**

---

## 15. Resume talking points

- Built an **end-to-end MLOps pipeline**: preprocessing → tuned XGBoost
  (Optuna, ROC-AUC ≈ 0.85) → MLflow tracking → FastAPI service → Docker → GitHub
  Actions CI/CD → live deployment.
- Engineered a **leak-proof scikit-learn Pipeline** so identical transformations
  run in training and serving.
- Added **schema-based input validation** (Pydantic) and **PSI-based data-drift
  monitoring** to keep predictions trustworthy in production.
- **19 automated tests** + lint run on every push via CI; the app is fully
  **containerized** and deployed on a free tier.

---

## Project layout

```
src/churn/      data.py · preprocess.py · train.py · drift.py · schemas.py · config.py
app/main.py     FastAPI service
tests/          pytest suite (data, preprocess, drift, api)
.github/workflows/ci.yml   CI/CD pipeline
Dockerfile · docker-compose.yml · Makefile · requirements*.txt
models/         trained pipeline + drift reference (committed)
```

See [AGENTS.md](AGENTS.md) for contributor/development conventions.
