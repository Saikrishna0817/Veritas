# 🛡️ AI Trust Forensics Platform v2.2

> **An experimental analyst-support platform for investigating risk signals consistent with adversarial data poisoning.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What is this?

**Data poisoning** is when an attacker secretly injects malicious training samples into an AI model's dataset. The model trains normally, passes all standard tests, gets deployed — and then silently causes harm. Our platform detects these attacks with a label check and a distribution-shift check against a trusted reference set (measured in `docs/benchmark-v2-findings.md`).

Built for the **Sustainable Development Goals (SDG) Hackathon** — specifically targeting SDG 3 (Good Health), SDG 9 (Infrastructure), and SDG 16 (Strong Institutions).

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **5-Layer Detection Pipeline** | Statistical shift → Spectral activation → Ensemble anomaly → Causal proof → Federated trust |
| **Attack Type Classification** | Automatically identifies: label flip, backdoor, clean label, gradient poisoning, boiling frog |
| **Proxy Impact Analysis** | Compares proxy-model outcomes with and without flagged rows; it does not prove an attack or deployment harm |
| **Model File Scanner** | Static scan (never loads the file) of pickle/joblib, PyTorch, safetensors, ONNX, NumPy and GGUF files for anything that can run code when loaded, in an isolated worker process |
| **Real Dataset Library** | Iris, Wine, Breast Cancer, Digits — with known-quantity poison injection for ground-truth validation |
| **SQLite Persistence** | All analysis results stored permanently and queryable via the History page |
| **Red Team Simulator** | Inject synthetic attacks and measure the platform's resilience in real time |
| **Blue Team SOC** | Security Operations Centre — threat level, HITL review queue, incident log, response playbooks |
| **Federated Trust** | Cosine similarity + EMA trust scoring for federated learning client safety |
| **Evidence Summaries** | Draft analyst summaries mapped to NIST AI RMF concepts; not evidence, attribution, or compliance certification |
| **Live WebSocket Feed** | Real-time event streaming for attack confirmations and defense actions |

---

## 🏗️ Architecture

```
├── backend/
│   └── app/
│       ├── detection/          # Veritas v2: label check + gated shift check (veritas2.py), pipeline.py adapter
│       ├── forensics/          # Attack reconstruction + narratives
│       ├── defense/            # Auto-defense + HITL + Red Team
│       ├── ingestion/          # CSV parsing, restricted sklearn loader (metadata only)
│       ├── scanner/            # Static model-file scanner (model_scan.py)
│       ├── demo/               # Synthetic + real public datasets
│       ├── api/routes/         # REST + WebSocket routes (auth, upload, models, datasets, reports, users, jobs)
│       ├── models/database.py  # SQLite persistence layer
│       └── jobs/               # In-process job queue for background uploads
│
└── frontend/
    └── src/
        └── pages/
            ├── Dashboard.jsx         # Live trust scores + radar chart
            ├── UploadPage.jsx        # Upload and analyse CSV files
            ├── ModelScanPage.jsx     # Scan model files
            ├── RealDatasetsPage.jsx  # Real dataset library
            ├── ForensicsPage.jsx     # Attack reconstruction details
            ├── AdminDashboard.jsx    # Admin console: users, red-team simulator
            ├── BlueTeamPage.jsx      # SOC — defense operations
            ├── FederatedPage.jsx     # Federated client trust
            ├── ReportsPage.jsx       # Regulatory evidence reports
            └── HistoryPage.jsx       # Past analysis results
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure secrets (the server refuses to start without them)
cp ../.env.example .env      # then fill in the three VERITAS_* values

# Start the server
uvicorn app.main:app --port 8001 --reload
```

The backend runs at `http://localhost:8001` — API docs at `http://localhost:8001/docs`

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend runs at `http://localhost:5173`

### Authentication and Docker

Copy [`.env.example`](.env.example) to `.env`, set a unique JWT secret and administrator
credentials, then run `docker compose up --build`. It documents every backend
and Vite environment variable. Obtain a JWT through
`POST /api/v1/auth/token` and sign in through the frontend. Apart from health
and token issuance, API and WebSocket endpoints require that JWT.

The server refuses to start when `VERITAS_JWT_SECRET` (min. 32 characters) or the admin
password (min. 8 characters) is missing or a well-known default, and no accounts are created
besides the administrator from `.env`. Create other users from the admin console.

Login attempts are limited to five per IP address per rolling minute (behind nginx the real
client IP is taken from `X-Forwarded-For`). The
current internal-deployment JWT storage decision and its security trade-off are
documented in [docs/decisions.md](docs/decisions.md).

---

## 🔬 The 5-Layer Detection Pipeline

Thresholds and scores are experimental analyst signals; their calibration and
evaluation requirements are documented in [docs/detection-calibration.md](docs/detection-calibration.md).
Current benchmark results and limitations are published in
[docs/model-card.md](docs/model-card.md); they show that the current detector
does not yet detect benchmark v1's seeded attacks at its selected threshold.

### Layer 1 — Statistical Shift Detection
Compares incoming data distribution to a clean baseline using:
- **KL Divergence** — information-theoretic distribution distance
- **Wasserstein Distance** — earth mover's distance between distributions
- **Mahalanobis Distance** — multivariate outlier detection accounting for feature correlations

### Layer 2 — Spectral Activation Analysis
Detects backdoor attacks via SVD (Singular Value Decomposition):
- A large **spectral gap** (S₀/S₁ ratio) indicates a backdoor subspace
- **KMeans** on PCA-reduced activations finds the trigger cluster

### Layer 3 — Ensemble Anomaly Detection
Four algorithms vote on each sample (≥2 votes = flagged):
- Isolation Forest · SGD One-Class SVM · Local Outlier Factor · DBSCAN

### Layer 4 — Causal Proof Engine
Inspired by Judea Pearl's Do-Calculus:
```
Causal Effect = Accuracy(without suspects) − Accuracy(with suspects)
```
Validated with: bootstrap 95% CI, placebo test, t-test (p < 0.05)

### Layer 5 — Federated Behavioral Trust
- Cosine similarity between client gradients and global gradient
- EMA trust score per client (α = 0.1)
- Auto-quarantine below 0.3 trust threshold

---

## 📡 API Endpoints (Selected)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/demo/run` | Run full analysis on demo data |
| `POST` | `/api/v1/analyze/upload` | Upload and analyse a CSV (`?background=true` returns a job id to poll at `GET /api/v1/jobs/{id}`) |
| `POST` | `/api/v1/analyze/model` | Statically scan a model file |
| `GET` | `/api/v1/datasets/real` | Real dataset catalog |
| `POST` | `/api/v1/redteam/simulate` | Inject attack + measure detection |
| `GET` | `/api/v1/blueteam/status` | SOC threat level + summary |
| `GET` | `/api/v1/blueteam/resilience` | Per-attack catch rate metrics |
| `GET` | `/api/v1/blueteam/playbook/{type}` | Incident response playbook |
| `GET` | `/api/v1/history` | SQLite-backed analysis history |
| `WS` | `/ws/v1/detection-stream` | Real-time event stream |

Full API docs: `http://localhost:8001/docs`

---

## 🌍 SDG Alignment

| SDG | Connection |
|-----|-----------|
| **SDG 3** — Good Health & Well-being | Prevents poisoned medical diagnostic AI from harming patients |
| **SDG 9** — Industry & Infrastructure | Provides security infrastructure for trustworthy AI deployment |
| **SDG 16** — Peace, Justice & Strong Institutions | Ensures AI used in governance/justice is tamper-proof |
| **SDG 17** — Partnerships | Secures federated learning between institutions without sharing raw data |

---

## 🛡️ Security

Legacy pickle uploads are deserialized only through a **restricted unpickler**. It rejects every unreviewed global before construction and permits only approved scikit-learn model classes and their required numeric containers. Pickle remains a legacy interoperability format; production deployments should prefer a non-executable model format or execute legacy parsing in an isolated worker. All API and WebSocket endpoints require a valid JWT except `/health`, `/`, and `/api/v1/auth/token`; the WebSocket sends its token in the first message, never in the URL. Analysis results are shared by all authenticated users (there is no per-user data isolation), so deploy for a single trusted team; see [docs/decisions.md](docs/decisions.md).

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML/Science | NumPy, SciPy, scikit-learn |
| Database | SQLite (WAL mode, thread-local connections) |
| Frontend | React 19, Vite, Tailwind CSS |
| Charts | Recharts |
| Icons | Lucide React |
| Real-time | WebSocket (native) |

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

Built with ❤️ for the Hackathon | AI Trust Forensics Platform v2.2
