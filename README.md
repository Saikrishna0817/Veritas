# 🛡️ AI Trust Forensics Platform

> **An experimental analyst-support platform for investigating risk signals consistent with adversarial data poisoning.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [What is Data Poisoning?](#what-is-data-poisoning)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [The 5-Layer Detection Pipeline](#the-5-layer-detection-pipeline)
- [API Endpoints](#api-endpoints)
- [SDG Alignment](#sdg-alignment)
- [Security Model](#security-model)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

**Data poisoning** is an attack where malicious samples are injected into a model's training data. The model trains normally, passes standard validation, gets deployed — and then silently produces harmful or biased outputs.

This platform provides a **forensic analyst workflow** for investigating such risk signals across five detection layers, generating causal evidence for manual review, and producing analyst-drafted summaries mapped to NIST AI RMF concepts.

**Important:** This is an **experimental research prototype**, not a certified detection system. Scores and thresholds are analyst-support signals only. The platform does **not** prove an attack occurred, attribute harm, or constitute compliance certification. See [Documentation](#documentation) for current benchmark limitations.

**Built for:** SDG 3 (Good Health), SDG 9 (Industry & Infrastructure), SDG 16 (Strong Institutions), SDG 17 (Partnerships).

---

## What is Data Poisoning?

An attacker secretly injects malicious training samples into an AI model's dataset. The model trains without error, passes all standard tests, and gets deployed — but then silently causes harm (misdiagnoses, manipulated financial decisions, corrupted autonomous systems).

Detection is hard because:
- Poisoned samples look statistically normal in isolation
- Backdoors activate only on specific trigger patterns
- Clean-label attacks don't even modify labels

This platform approaches the problem from five independent angles and correlates the results into a **risk narrative for human analysts to investigate**.

---

## Features

| Feature | Description |
|---|---|
| **5-Layer Detection Pipeline** | Statistical shift → Spectral activation → Ensemble anomaly → Causal proof → Federated trust |
| **Attack Type Classification** | Identifies label flip, backdoor, clean label, gradient poisoning, and boiling frog patterns |
| **Proxy Impact Analysis** | Compares proxy-model outcomes with and without flagged rows. **Does not** prove attack or deployment harm. |
| **Model Scanner** | Uploads scikit-learn `.pkl` models through a restricted legacy parser and scans parameters for anomalies |
| **Real Dataset Library** | Iris, Wine, Breast Cancer, Digits — with known-quantity poison injection for ground-truth validation |
| **SQLite Persistence** | All analysis results stored and queryable via the History page |
| **Red Team Simulator** | Inject synthetic attacks and measure resilience in real time |
| **Blue Team SOC** | Threat level, HITL review queue, incident log, response playbooks |
| **Federated Trust** | Cosine similarity + EMA trust scoring for federated learning clients |
| **Analyst Summaries** | Draft summaries mapped to NIST AI RMF concepts (not certification) |
| **Live WebSocket Feed** | Real-time streaming of detection and defense events |

---

## Architecture

```
├── backend/
│   └── app/
│       ├── detection/          # 5-layer detection pipeline
│       │   ├── layer1_statistical.py   # KL Divergence, Wasserstein, Mahalanobis
│       │   ├── layer2_spectral.py      # SVD spectral gap + KMeans backdoor detection
│       │   ├── layer3_ensemble.py      # IsolationForest + SVM + LOF + DBSCAN voting
│       │   ├── layer4_causal.py        # Counterfactual causal proof engine
│       │   └── layer5_federated.py     # Cosine similarity + EMA trust scoring
│       ├── forensics/          # Attack reconstruction + narratives
│       ├── defense/            # Auto-defense + HITL + Red Team
│       ├── ingestion/          # CSV + Model (.pkl) parsing
│       ├── demo/               # Synthetic + real public datasets
│       ├── db/                 # SQLite persistence layer
│       └── api/routes.py       # 29 REST endpoints + WebSocket
│
└── frontend/
    └── src/
        └── pages/
            ├── Dashboard.jsx         # Live trust scores + radar chart
            ├── UploadPage.jsx        # Upload and analyse CSV files
            ├── ModelScanPage.jsx     # Scan .pkl models
            ├── RealDatasetsPage.jsx  # Real dataset library
            ├── ForensicsPage.jsx     # Attack reconstruction details
            ├── RedTeamPage.jsx       # Attack simulation
            ├── BlueTeamPage.jsx      # SOC — defense operations
            ├── FederatedPage.jsx     # Federated client trust
            ├── ReportsPage.jsx       # Regulatory evidence reports
            └── HistoryPage.jsx       # Past analysis results
```

**Design principle:** detection layers, ingestion adapters, persistence, and the API layer are independent modules. Each detection layer can be run in isolation, and results are correlated at the API layer rather than being hard-wired together.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (optional) Docker for containerised deployment

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --port 8001 --reload
```

Backend: `http://localhost:8001`
API docs: `http://localhost:8001/docs`

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Frontend: `http://localhost:5173`

### 3. Authentication & Docker

1. Copy `.env.example` → `.env`.
2. Set a unique `JWT_SECRET` and administrator credentials.
3. Run:

```bash
docker compose up --build
```

4. Obtain a JWT via `POST /api/v1/auth/token`.
5. Sign in through the frontend.

> **Auth requirements:** All API and WebSocket endpoints require a valid JWT, except `/health`, `/`, and `/api/v1/auth/token`.

> **Rate limiting:** Login attempts are limited to five per IP address per rolling minute.

The current JWT storage decision and its security trade-off are documented in [`docs/decisions.md`](docs/decisions.md).

---

## The 5-Layer Detection Pipeline

Each layer approaches the problem from an independent angle. Correlation across layers produces the final risk score and attack-type classification.

> ⚠️ **Thresholds and scores are experimental analyst signals.** Calibration and evaluation requirements are documented in [`docs/detection-calibration.md`](docs/detection-calibration.md). Current benchmark results are published in [`docs/model-card.md`](docs/model-card.md) — they show the current detector **does not yet detect benchmark v1's seeded attacks at its selected threshold.**

### Layer 1 — Statistical Shift Detection

Compares incoming data distribution against a clean baseline:

| Metric | Purpose |
|---|---|
| **KL Divergence** | Information-theoretic distribution distance |
| **Wasserstein Distance** | Earth mover's distance between distributions |
| **Mahalanobis Distance** | Multivariate outlier detection accounting for feature correlations |

### Layer 2 — Spectral Activation Analysis

Detects backdoor attacks via SVD:

- A large **spectral gap** (S₀/S₁ ratio) indicates a backdoor subspace.
- **KMeans** on PCA-reduced activations finds the trigger cluster.

### Layer 3 — Ensemble Anomaly Detection

Four algorithms vote on each sample (≥2 votes = flagged):

- Isolation Forest
- SGD One-Class SVM
- Local Outlier Factor
- DBSCAN

### Layer 4 — Causal Proof Engine

Inspired by Judea Pearl's Do-Calculus:

```
Causal Effect = Accuracy(without suspects) − Accuracy(with suspects)
```

Validated with:

- Bootstrap 95% confidence interval
- Placebo test
- t-test (p < 0.05)

### Layer 5 — Federated Behavioral Trust

- Cosine similarity between client gradients and global gradient
- EMA trust score per client (α = 0.1)
- Auto-quarantine below 0.3 trust threshold

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/demo/run` | Run full analysis on demo data |
| `POST` | `/api/v1/analyze/upload` | Upload and analyse a CSV |
| `POST` | `/api/v1/analyze/model` | Scan a `.pkl` model file |
| `GET` | `/api/v1/datasets/real` | Real dataset catalog |
| `POST` | `/api/v1/redteam/simulate` | Inject attack + measure detection |
| `GET` | `/api/v1/blueteam/status` | SOC threat level + summary |
| `GET` | `/api/v1/blueteam/resilience` | Per-attack catch rate metrics |
| `GET` | `/api/v1/blueteam/playbook/{type}` | Incident response playbook |
| `GET` | `/api/v1/history` | SQLite-backed analysis history |
| `WS` | `/ws/v1/detection-stream` | Real-time event stream |

Full interactive API docs: `http://localhost:8001/docs`

---

## SDG Alignment

| SDG | Connection |
|---|---|
| **SDG 3 — Good Health & Well-being** | Prevents poisoned medical diagnostic AI from harming patients |
| **SDG 9 — Industry & Infrastructure** | Provides security infrastructure for trustworthy AI deployment |
| **SDG 16 — Peace, Justice & Strong Institutions** | Ensures AI used in governance/justice is tamper-resistant |
| **SDG 17 — Partnerships** | Secures federated learning between institutions without sharing raw data |

---

## Security Model

### Pickle Deserialization

Legacy `.pkl` uploads are deserialized only through a **restricted unpickler**:

- Rejects every unreviewed global before construction.
- Permits only approved scikit-learn model classes and their required numeric containers.

### Recommendation

Pickle is a legacy interoperability format. **Production deployments should** prefer a non-executable model format or execute legacy parsing in an isolated worker.

### Authentication

- All API and WebSocket endpoints require a valid JWT.
- Exceptions: `/health`, `/`, `/api/v1/auth/token`.
- Login attempts: 5 per IP per rolling minute.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, FastAPI, Uvicorn |
| ML / Science | NumPy, SciPy, scikit-learn |
| Database | SQLite (WAL mode, thread-local connections) |
| Frontend | React 18, Vite, Tailwind CSS |
| Charts | Recharts |
| Icons | Lucide React |
| Real-time | Native WebSocket |
| Deployment | Docker Compose |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/detection-calibration.md`](docs/detection-calibration.md) | Threshold calibration and evaluation requirements |
| [`docs/model-card.md`](docs/model-card.md) | Current benchmark results and limitations |
| [`docs/decisions.md`](docs/decisions.md) | JWT storage decision and security trade-offs |

---

## License

MIT — see [LICENSE](LICENSE)

---

**Status:** Experimental research prototype. Not certified for production detection. All scores and thresholds are analyst-support signals requiring human review.

