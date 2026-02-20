# 🛡️ AI Trust Forensics Platform v2.2

> **Detecting, proving, and explaining adversarial data poisoning attacks on AI/ML systems — in real time.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What is this?

**Data poisoning** is when an attacker secretly injects malicious training samples into an AI model's dataset. The model trains normally, passes all standard tests, gets deployed — and then silently causes harm. Our platform detects these attacks using a 5-layer forensic pipeline, proves the harm causally, and generates regulatory evidence.

Built for the **Sustainable Development Goals (SDG) Hackathon** — specifically targeting SDG 3 (Good Health), SDG 9 (Infrastructure), and SDG 16 (Strong Institutions).

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **5-Layer Detection Pipeline** | Statistical shift → Spectral activation → Ensemble anomaly → Causal proof → Federated trust |
| **Attack Type Classification** | Automatically identifies: label flip, backdoor, clean label, gradient poisoning, boiling frog |
| **Causal Proof Engine** | Mathematically *proves* harm using counterfactual analysis, bootstrap CI, and placebo tests |
| **Model Scanner** | Upload `.pkl` scikit-learn models and scan their parameters for signs of poisoning |
| **Real Dataset Library** | Iris, Wine, Breast Cancer, Digits — with known-quantity poison injection for ground-truth validation |
| **SQLite Persistence** | All analysis results stored permanently and queryable via the History page |
| **Red Team Simulator** | Inject synthetic attacks and measure the platform's resilience in real time |
| **Blue Team SOC** | Security Operations Centre — threat level, HITL review queue, incident log, response playbooks |
| **Federated Trust** | Cosine similarity + EMA trust scoring for federated learning client safety |
| **Regulatory Reports** | NIST AI RMF and EU AI Act compliant evidence packages |
| **Live WebSocket Feed** | Real-time event streaming for attack confirmations and defense actions |

---

## 🏗️ Architecture

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

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

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

---

## 🔬 The 5-Layer Detection Pipeline

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
| `POST` | `/api/v1/analyze/upload` | Upload and analyse a CSV |
| `POST` | `/api/v1/analyze/model` | Scan a `.pkl` model file |
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

The model scanner uses **pre-execution opcode scanning** — suspicious `.pkl` files are scanned at the bytecode level before any code executes. Only whitelisted scikit-learn classes are allowed.

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, FastAPI, Uvicorn |
| ML/Science | NumPy, SciPy, scikit-learn |
| Database | SQLite (WAL mode, thread-local connections) |
| Frontend | React 18, Vite, Tailwind CSS |
| Charts | Recharts |
| Icons | Lucide React |
| Real-time | WebSocket (native) |

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

Built with ❤️ for the Hackathon | AI Trust Forensics Platform v2.2
