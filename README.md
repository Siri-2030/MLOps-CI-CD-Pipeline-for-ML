# CI/CD Pipeline for Machine Learning using GitHub Actions
### Wine Quality Classification — End-to-End MLOps Project

---

## 1. Project Title

**CI/CD Pipeline for Machine Learning using GitHub Actions**
A complete, working MLOps system built around a wine quality classification model.

---

## 2. Project Overview

This project demonstrates a full MLOps lifecycle: a machine learning model is trained on
versioned data, tracked with MLflow, validated against quality thresholds, containerized
with Docker, served via FastAPI, monitored with Prometheus and Grafana, and automatically
built/deployed through a GitHub Actions CI/CD pipeline — with every stage triggered
automatically by Git activity (code or data changes) and gated by real automated checks.

---

## 3. Problem Statement

Manually retraining, validating, and redeploying machine learning models is slow and
error-prone. Without automation, there's no guarantee that a newly trained model actually
performs well before it reaches production, and no consistent, reproducible way to package
and serve it. This project solves that by building a pipeline that automatically tests,
validates, and deploys a model — and **blocks deployment** if the model doesn't meet a
minimum performance bar.

---

## 4. Objectives

- Version source code (Git/GitHub) and datasets (DVC) separately and correctly
- Track every training run's parameters, metrics, and model artifacts (MLflow)
- Automatically test, validate data, train, and validate model quality on every push
- Enforce a real quality gate that blocks deployment of underperforming models
- Package the trained model as a Docker image and serve it via FastAPI
- Monitor the live API with Prometheus and visualize it in Grafana
- Automate the entire flow end-to-end with GitHub Actions (CI + CD)

---

## 5. Technologies Used

| Category | Tool |
|---|---|
| Source control | Git, GitHub |
| Dataset versioning | DVC (remote: DagsHub) |
| Experiment tracking | MLflow (SQLite backend) |
| ML | Python 3.11, scikit-learn (Random Forest) |
| Testing | pytest, FastAPI TestClient |
| Linting | ruff |
| Model serving | FastAPI, Uvicorn, Pydantic |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry (GHCR) |
| Monitoring | Prometheus, Grafana |

---

## 6. Architecture

```
Developer → Git/GitHub → GitHub Actions (CI)
  → Lint → Tests → DVC Pull → Data Validation → Train → MLflow
  → Model Validation Gate --[FAIL]--> Pipeline stops, deployment blocked
                          --[PASS]--> Docker Build
                                        → GitHub Actions (CD)
                                          → Push to GHCR → Deploy → Health Check
                                            → FastAPI (serving)
                                              → Prometheus (scrapes /metrics)
                                                → Grafana (dashboards)
```

CI and CD are two separate workflows. CD triggers only after CI completes successfully
(`workflow_run` trigger) — if CI's model validation step fails, CD never starts at all.

---

## 7. Folder Structure

```
MLOps_CICD_Project/
├── .github/workflows/       # ci.yml, cd.yml
├── data/raw/                # DVC-tracked dataset
├── data/processed/          # generated, git-ignored
├── src/                     # config, preprocessing, train, model_validation
├── app/                     # FastAPI application (main.py)
├── models/                  # trained model.pkl, metrics.json (git-ignored)
├── tests/                   # test_data.py, test_model.py, test_validation.py, test_api.py
├── monitoring/               # prometheus.yml, grafana/ (datasource + dashboard)
├── scripts/                  # validate_data.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt / requirements-dev.txt
├── params.yaml
├── .gitignore / .dockerignore
└── README.md
```

---

## 8. ML Workflow

**Dataset:** UCI Wine Quality (Red) — 1,599 physicochemical samples (later expanded to
1,619 in Demo 3), 11 numeric features, target `quality` (integer score 0–10).

**Why this dataset:** small, clean, tabular, real downloadable file (good for demonstrating
DVC), well-known enough that reviewers understand it immediately, without needing heavy
preprocessing that would distract from the MLOps focus.

**Target framing:** quality score bucketed into 3 classes based on the dataset's actual
distribution (inspected before choosing boundaries, not guessed):
- **low**: quality ≤ 5 (744 samples, 46.5%)
- **medium**: quality = 6 (638 samples, 39.9%)
- **high**: quality ≥ 7 (217 samples, 13.6%)

**Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=10`,
`random_state=42`) — chosen for simplicity and reliability, keeping the project's focus
on MLOps rather than ML complexity.

**Measured performance** (on the original 1,599-row dataset, 80/20 split):

| Metric | Value |
|---|---|
| Accuracy | 0.7469 |
| Precision (macro) | 0.7314 |
| Recall (macro) | 0.6953 |
| F1 Score (macro) | 0.7096 |

---

## 9. DVC Workflow

```
Git      → tracks code and a small .dvc pointer file (hash reference)
DVC      → tracks the actual dataset bytes, stored on a remote (DagsHub)
MLflow   → tracks experiment runs, metrics, and model artifacts
```

- `dvc init` initializes DVC tracking in the repo
- `dvc add data/raw/winequality-red.csv` hashes the file, moves it into DVC's cache,
  and creates `winequality-red.csv.dvc` — the small pointer file Git actually commits
- `dvc remote add origin <DagsHub URL>` configures where the real data lives
- `dvc push` / `dvc pull` upload/download the actual dataset bytes
- Credentials (DagsHub username/token) are stored in `.dvc/config.local`, which is
  git-ignored — never committed, and provided to CI via GitHub Secrets

This was demonstrated live in **Demo 3**: the dataset was modified (1,599 → 1,619 rows),
re-tracked with `dvc add`, pushed to the remote, and CI automatically pulled the new
version and retrained on it.

---

## 10. MLflow Workflow

Every training run is wrapped in an `mlflow.start_run()` context that logs:
- **Parameters:** model_type, n_estimators, max_depth, random_state, test_size
- **Metrics:** accuracy, precision, recall, f1_score
- **Artifacts:** the trained model itself (via `mlflow.sklearn.log_model`)

MLflow uses a local SQLite backend (`sqlite:///mlflow.db`) rather than the legacy
filesystem store, per MLflow 3.x's current recommendation. Runs are viewable via:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**How it fits into CI/CD:** every automated training run in CI logs to MLflow the same
way a manual run does, so even fully automated pipeline runs leave a searchable,
comparable experiment history — not just a final pass/fail signal.

---

## 11. CI Pipeline

Workflow: `.github/workflows/ci.yml`
Triggers: push/PR to `main`, path-filtered to files that actually affect the pipeline
(`src/**`, `app/**`, `tests/**`, `data/**`, `*.dvc`, `params.yaml`, `Dockerfile`,
requirements files, workflow files).

**Stages (in order):**
1. Checkout
2. Python 3.11 setup
3. Install dependencies (`requirements-dev.txt`)
4. Lint (`ruff check`)
5. Configure DVC remote credentials (from GitHub Secrets)
6. `dvc pull` — fetch the current dataset version
7. Dataset validation (`scripts/validate_data.py`)
8. **Train model** (must precede tests, since API tests load the trained model)
9. Run full pytest suite (44 tests)
10. **Model validation gate** — blocks the pipeline if thresholds aren't met
11. Docker build (only reached if everything above passed)

---

## 12. CD Pipeline

Workflow: `.github/workflows/cd.yml`
Trigger: `workflow_run`, firing only when `CI Pipeline` completes with `conclusion == success`.

**Stages:**
1. Checkout the exact commit CI validated (`head_sha`)
2. Re-train (reproducing the identical model CI validated, via fixed `random_state`)
3. Log in to GHCR using the auto-provided `GITHUB_TOKEN`
4. Build and tag the Docker image (both `latest` and the commit SHA)
5. Push to GHCR
6. Deploy a container locally and run a real health check (`/health`)
7. Verify `/predict` responds correctly
8. Report success/failure; clean up the test container

**Why rebuild instead of passing artifacts between jobs:** simpler and fully reproducible
given our fixed random seed — avoids the added complexity of GitHub Actions artifact
upload/download between separate workflow runs, at negligible extra CI time cost for a
small dataset like this.

---

## 13. Docker Architecture

- **Dockerfile:** `python:3.11-slim` base → installs `requirements.txt` (production deps
  only) → copies `app/`, `src/`, `params.yaml`, and the trained `models/model.pkl` →
  runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **`.dockerignore`** excludes `.venv/`, `.git/`, `data/`, `mlruns/`, tests, and dev-only
  files to keep the image lean and build context fast
- **docker-compose.yml** runs three services on a shared bridge network (`wine-net`):
  `api` (FastAPI), `prometheus`, `grafana` — enabling them to reach each other by
  service name (e.g. Prometheus scrapes `api:8000`)

---

## 14. FastAPI Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | App info, model load status, startup timestamp |
| `/health` | GET | Health check — returns 503 if model isn't loaded |
| `/predict` | POST | Accepts 11 wine features, returns predicted class + model version |
| `/metrics` | GET | Prometheus-format metrics (requests, predictions, errors, latency) |

Input validation is handled via Pydantic — malformed or incomplete requests return
`422 Unprocessable Entity` rather than crashing the server.

---

## 15. Prometheus Monitoring

`monitoring/prometheus.yml` configures Prometheus to scrape `api:8000/metrics` every 5
seconds. Metrics exposed:
- `app_request_count_total`
- `app_prediction_count_total`
- `app_error_count_total`
- `app_request_latency_seconds_avg`
- `app_status_code_count{code="..."}`

Verified live: Prometheus's own UI (`http://localhost:9090/targets`) shows the
`wine-quality-api` target as **UP**, confirming successful scraping.

---

## 16. Grafana Dashboard

Auto-provisioned via `monitoring/grafana/datasource.yml` (Prometheus data source) and
`monitoring/grafana/dashboards/` (dashboard JSON + provisioning config) — no manual
import needed on stack startup.

**Panels:** Total Requests, Requests per Second, Prediction Requests, Error Count,
Average Request Latency, API Health/Status Codes (pie chart).

**Verified live** (Demo 5): after sending 15 prediction requests, the dashboard correctly
showed 15 prediction requests, 0 errors, and all 200 status codes — proving the full
scrape → store → visualize chain works end-to-end.

---

## 17. Installation Instructions

```powershell
git clone https://github.com/Siri-2030/MLOps-CI-CD-Pipeline-for-ML.git
cd MLOps-CI-CD-Pipeline-for-ML
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Pull the dataset:
```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
dvc pull
```

---

## 18. Local Execution

```powershell
# Train the model
python -m src.train

# Run tests
pytest -v

# Validate the model against thresholds
python -m src.model_validation

# Run the API locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or run the full monitoring stack
docker compose up --build -d
```

Access points once the Compose stack is running:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## 19. GitHub Actions Explanation

GitHub Actions runs workflows defined in `.github/workflows/*.yml` in response to repo
events. A **workflow** contains one or more **jobs**; each job runs on a fresh virtual
machine (`runs-on: ubuntu-latest`) and consists of ordered **steps**. If any step fails,
the job stops immediately — later steps (like Docker build) never execute. This project
uses two workflows (`ci.yml`, `cd.yml`) chained via the `workflow_run` trigger, so CD is
structurally incapable of running unless CI succeeded first.

---

## 20. GitHub Secrets

Two secrets are configured under repo **Settings → Secrets and variables → Actions**:
- `DAGSHUB_USERNAME` — DagsHub account username
- `DAGSHUB_TOKEN` — DagsHub personal access token (used for `dvc pull`/`push` in CI/CD)

`GITHUB_TOKEN` (used for GHCR authentication) is provided automatically by GitHub — no
manual setup required. No secret ever appears in workflow YAML files, source code, or
commit history.

---

## 21. Deployment Instructions

Primary deployment target: **local Docker Compose** — chosen for reliability during
academic demonstration (no dependency on external cloud uptime or free-tier limits).

```powershell
docker compose up --build -d
```

The CD pipeline additionally deploys and health-checks a container automatically inside
GitHub Actions itself on every successful CI run, proving the image is genuinely
deployable, not just buildable.

---

## 22. Testing

44 tests across 4 files, using pytest and FastAPI's `TestClient`:

| File | Tests | Covers |
|---|---|---|
| `test_data.py` | 15 | Dataset loading, bucketing logic, preprocessing, train/test split |
| `test_model.py` | 9 | Training, prediction, evaluation metrics |
| `test_validation.py` | 6 | Validation gate pass/fail logic |
| `test_api.py` | 14 | All 4 endpoints, valid/invalid input handling |

All 44 pass in both local and CI (Ubuntu, fresh environment) runs.

---

## 23. Failure Handling

Demonstrated and verified (not just described):

- **Test failure** → pipeline stops, no training/deployment occurs (enforced by GitHub
  Actions' step-ordering)
- **Model validation failure (Demo 4)** → `min_accuracy` temporarily raised to 0.99;
  CI's validation step failed with exit code 1; CD never triggered at all, since it
  only fires on CI success
- **Docker build failure** → would block CD identically (build step precedes push/deploy)
- **Health check failure** → CD's health-check step uses `curl -f`, which fails the job
  if the container doesn't respond correctly, preventing a "successful" deployment
  report for a broken container

---

## 24. Screenshots / Evidence Checklist

See `evidence_checklist.md` — a 20-section checklist covering every phase, command, and
demo, with space to record actual measured metrics rather than assumed values.

---

## 25. Future Enhancements

- Add a real cloud deployment target (e.g. Render) as a documented alternative to local
  Compose, for remote demo access
- Expand the model comparison (try Gradient Boosting alongside Random Forest, log both
  to MLflow, compare via the UI)
- Add model versioning/promotion via MLflow's Model Registry
- Add Grafana alerting rules (e.g. notify if error rate exceeds a threshold)
- Add integration tests that spin up the full Compose stack in CI itself

---

## 26. Conclusion

This project demonstrates a genuinely complete, working, and verified MLOps CI/CD
pipeline — every stage from dataset versioning through automated deployment and
monitoring has been built and independently verified with real command output. Five
demo scenarios (initial pipeline, code-change trigger, dataset-change trigger, validation
failure blocking deployment, and live monitoring) confirm the system behaves correctly
under both normal and failure conditions, fulfilling the project's core goal: **Git +
GitHub + DVC + MLflow + GitHub Actions + Docker + FastAPI + Prometheus + Grafana =
End-to-End MLOps CI/CD.**
