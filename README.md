# Veritas

Veritas checks a training dataset for poisoning before you train a model on it. You upload a table, or a zip of
images, audio or video in class folders, ideally with a reference set you trust. Veritas measures how far the new
data departs from that reference and lists the rows or files to review first. A verdict is a calibrated false-alarm
level ("evidence at the 1-in-10,000 level"), not proof that the data is poisoned. Veritas also scans model files for
code that would run when the file is loaded.

Product context and principles: [docs/PRODUCT.md](docs/PRODUCT.md). Accuracy, limits and data handling:
[docs/model-card.md](docs/model-card.md).

---

## What you can check

| Data | Upload | Size limit | What is read |
|---|---|---|---|
| Table | `.csv` with a label column | 200 MB | Text columns with up to 50 distinct values are checked as categories. |
| Images | `.zip`, one folder per class (`cat/001.png`) | 500 MB | Up to 6,000 images, resized to 32×32. |
| Audio | `.zip` of PCM WAV files, one folder per class (`dog/001.wav`) | 500 MB | Clips mixed to mono and padded or trimmed to one length. |
| Video | `.zip` of mp4, avi, mov, mkv or webm files, one folder per class | 500 MB | 8 frames per clip. |

**The reference.** A verdict needs data you trust:

- a separate reference file (CSV) or zip;
- for tables, the rows before a cutoff in a column you name;
- or, if you opt in, the first 70% of a table or a random 70% of each class.

With no reference, which is the default, every row or item is checked against the others. The result is a
screening: one ranked list per check, and no verdict.

**A result contains:**

- the verdict, as a false-alarm level:

  | Verdict | False-alarm level |
  |---|---|
  | `CONFIRMED_POISONED` | 1 in 10,000 |
  | `SUSPICIOUS` | 1 in 100 |
  | `LOW_RISK` | 1 in 20 |
  | `CLEAN` | below 1 in 20 |
  | `SCREENED` | no verdict; screening without a reference |

- the evidence from each check, as −log₁₀ of a p-value measured against your reference;
- the items to review: for tables, rows with their line in your file and the cells most to blame, as you wrote
  them; for images, audio and video, files with their class and the check that flagged them;
- for tables, which of six benchmarked attacks the suspicious rows look most like;
- the limits of that check, shown next to the result. For example, images are checked at 32×32, so a smaller trigger
  can vanish, and audio and video do not detect triggers added to correctly labelled clips. The full list is in
  `frontend/src/components/results.jsx` (`KINDS`) and [docs/model-card.md](docs/model-card.md#known-limits).

**The checks.**

- **Tables (`backend/app/detection/veritas2.py`):** a cross-fitted label check, a distribution-shift check and a
  column-consistency check. The shift and column checks join the row ranking only when their own evidence is strong.
- **Images, audio and video (`backend/app/detection/veritas_mm.py`):** a shared set of checks:
  - label;
  - content shift;
  - pixel statistics;
  - near-duplicates;
  - shared patch;
  - shared faint pattern;
  - colour noise;
  - a within-class pattern check, used for images and audio, not video.

  Each type is first turned into a 32×32 view and an embedding:
  - images: a ResNet-18 embedding;
  - audio: a log-mel spectrogram view and a HuBERT embedding;
  - video: the mean of 8 frames.

### Model-file scanner

The scanner answers one question: can loading this file run code? It never loads or runs the file:

- Pickles (including the ones inside PyTorch and joblib files) are read opcode by opcode.
- Every import they would make is checked against a deny list and a known-safe list.
- safetensors, GGUF, ONNX and plain `.npy` are only checked for valid structure.

It accepts `.pkl`, `.pickle`, `.joblib`, `.pt`, `.pth`, `.bin`, `.ckpt`, `.safetensors`, `.onnx`, `.npy`, `.npz`,
`.gguf`, `.h5`, `.keras` and `.zip`, up to 200 MB. The scan runs in a separate process with resource limits.

| Verdict | Meaning |
|---|---|
| `DANGEROUS` | The file imports something that can run code or touch the system. |
| `REVIEW` | Unknown imports, or content the scanner can't parse. A person should look. |
| `NO_CODE_EXECUTION_FOUND` | A pickle whose imports are all known data containers. This is not a guarantee of safety. |
| `SAFE_FORMAT` | A format that can't contain code, and its structure checks passed. |

What it does not do:

- It does not judge what a model has learned, so a backdoor in the weights is out of scope.
- It does not prove a file is safe.
- Its malicious test files were written by the same person who wrote the scanner. See
  [docs/scanner-benchmark.md](docs/scanner-benchmark.md) and the scanner section of the model card.

---

## Accuracy

Accuracy figures for every check are in these documents, which say how they were measured:

- [docs/model-card.md](docs/model-card.md): every check, with its known limits
- [docs/benchmark-v2-findings.md](docs/benchmark-v2-findings.md): tables
- [docs/image-benchmark-findings.md](docs/image-benchmark-findings.md): images
- [docs/audio-benchmark-findings.md](docs/audio-benchmark-findings.md): audio
- [docs/video-benchmark-findings.md](docs/video-benchmark-findings.md): video
- [docs/scanner-benchmark.md](docs/scanner-benchmark.md): model-file scanner

The in-app **Benchmarks** page summarises the held-out results. The numbers are not repeated here, so that they
can't drift from those documents.

---

## Frontend routes

Routes are defined in `frontend/src/App.jsx`.

| Route | Page | For |
|---|---|---|
| `/login`, `/admin/login` | `LoginPage` | Sign in; the admin variant asks for the admin role. |
| `/signup`, `/verify-email`, `/forgot-password`, `/reset-password` | `AccountFlowPage` | Create an account, confirm the email, reset a password. |
| `/` | `Dashboard` | Overview: the latest result for each data type, and what a result means. |
| `/upload` | `UploadPage` | Check data: choose the data type, the file and the reference, then read the result. |
| `/history`, `/history/:id` | `HistoryPage` | Every dataset you have checked, filtered by type; open one for its full result. |
| `/model-scan` | `ModelScanPage` | Scan a model file and see past scans. |
| `/reports` | `ReportsPage` | A printable record of one analysis, with response steps when something was found. |
| `/benchmarks` | `BenchmarksPage` | Held-out results per data type and what no check catches yet. |
| `/admin` | `AdminDashboard` | Admin only: users, defence actions and review queue, audit log, analysis records. |
| `/forensics`, `/blue-team`, `/real-datasets` | redirects | Older links: they go to `/history`, `/reports` and `/upload`. |

Signed-in users see Check data, History, Scanner, Reports and Benchmarks in the navigation. Admins see the Admin
console and History.

---

## Repository layout

```
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routes (api/routes/*.py) and shared dependencies
│   │   ├── core/            # settings, security (session cookie, CSRF), quotas, storage, mail, observability
│   │   ├── defense/         # defence actions and the human review queue
│   │   ├── demo/            # synthetic sample batch and the small public reference datasets
│   │   ├── detection/       # veritas2 (tables), veritas_mm (images/audio/video), embeddings, attack profile
│   │   ├── forensics/       # attack classification, injection pattern, blast radius
│   │   ├── ingestion/       # CSV and zip readers (images, audio, video) with size and zip-bomb guards
│   │   ├── jobs/            # background jobs: in-process, or Redis/RQ workers (separate queue for untrusted files)
│   │   ├── models/          # database layer and SQL migrations (SQLite or Postgres)
│   │   ├── scanner/         # static model-file scanner
│   │   ├── utils/
│   │   └── main.py
│   ├── research/            # benchmark harnesses and studies behind docs/*-findings.md
│   ├── tests/               # backend tests (pytest)
│   ├── requirements.txt
│   ├── requirements-images.txt   # torch, torchvision, torchaudio, pillow for image/audio/video checks
│   └── Dockerfile
├── cli/veritas_cli.py       # command-line client: scan-model, check-data (standard library only)
├── docs/                    # product context, model card, benchmark findings, decisions, runbook, legal
├── e2e/                     # Playwright end-to-end tests against a running stack
├── frontend/                # React app (see frontend/README.md)
├── loadtest/k6.js           # k6 load test
├── scripts/                 # database backup and restore drill
├── tests/fixtures/          # shared test files
├── docker-compose.yml       # Postgres, Redis, API, analysis worker, isolated media worker, nginx frontend
└── pytest.ini
```

---

## Quick start

### Prerequisites

- Python 3.12. The backend image is `python:3.12-slim`, and the pinned numpy needs it.
- Node.js 20+.
- For image, audio and video checks: `backend/requirements-images.txt`, plus ffmpeg for video.

### Configuration

The backend reads `.env` from `backend/` or the repository root; real environment variables win. It refuses to
start without these settings:

| Variable | Requirement |
|---|---|
| `VERITAS_JWT_SECRET` | Random, at least 32 characters. |
| `VERITAS_ADMIN_USERNAME` | Any username; this admin account is created at startup. |
| `VERITAS_ADMIN_PASSWORD` | Not a known default, at least 8 characters. |

Every variable, with defaults, is listed in [docs/environment.md](docs/environment.md).

Docker needs `POSTGRES_PASSWORD` too. `.env.example` lists every setting `docker-compose.yml` passes through (mail,
Turnstile, S3 storage, CORS, the public URL and the rest) with placeholder values: `cp .env.example .env` and fill it in.

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
# optional, for image/audio/video checks (CPU-only torch):
# pip install -r requirements-images.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
uvicorn app.main:app --port 8001 --reload
```

The API runs at `http://localhost:8001`. Interactive docs are at `http://localhost:8001/docs`. Without
`VERITAS_DATABASE_URL` it uses SQLite. Without `VERITAS_REDIS_URL`, background jobs run inside the API process.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173`. Vite proxies `/api` and `/ws` to `localhost:8001`. Set `VITE_API_PROXY`
to use another address.

Signed-out visitors at `/` are sent to the landing page at `/welcome/`, as nginx serves it in production. Locally,
run the landing page's own dev server too; the app's dev server forwards `/welcome` to it (`VITE_LANDING_PROXY` to
change the address):

```bash
cd frontend/landing && npm install && npx vite --port 5180
```

### Docker

```bash
docker compose up --build
```

This starts:

- Postgres and Redis;
- the API;
- a worker for CSV jobs;
- an isolated media worker for image, audio, video and model-file jobs (no internet, read-only filesystem, capped
  resources);
- nginx serving the frontend.

### Sign-in and sessions

Sign in through the app, or with `POST /api/v1/auth/token`. The session is an httpOnly cookie (`veritas_session`).
Every request that changes something must also echo the `veritas_csrf` cookie in an `X-CSRF-Token` header. The CLI
uses API keys instead (`Authorization: Bearer vk_…`).

- Login is limited to 5 attempts per IP address per minute, and 10 per username per 15 minutes.
- Self-service sign-up with email confirmation is on unless `VERITAS_SIGNUP_ENABLED=false`.
- A user sees only their own analyses; an admin sees everyone's.

---

## API

All paths below are under `/api/v1` unless noted; the full schema is at `/docs`. Except for health and the
sign-in routes, every route needs a session or an API key.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health`, `/` (no prefix) | Health check and service info |
| `POST` | `/auth/token`, `/auth/logout` | Sign in (sets the session cookie), sign out |
| `GET` | `/auth/me` | The signed-in user |
| `POST` | `/auth/signup`, `/auth/verify-email`, `/auth/resend-verification` | Create an account and confirm its email |
| `POST` | `/auth/password-reset/request`, `/auth/password-reset/confirm` | Reset a password by email |
| `GET` | `/account/usage` | Today's usage and limits |
| `GET`, `POST`, `DELETE` | `/account/api-keys`, `/account/api-keys/{key_id}` | List, create and revoke API keys |
| `DELETE` | `/account/data` | Erase your stored results, scans, defence actions, review cases and usage counters; the account stays |
| `POST` | `/account/delete` | Delete your account |
| `POST` | `/analyze/upload` | Check a CSV (`?background=true` returns a job id) |
| `POST` | `/analyze/images`, `/analyze/audio`, `/analyze/video` | Check a zip; always a background job |
| `GET` | `/analyze/upload/latest`, `/analyze/upload/{dataset_id}` | Your latest upload result, or one of yours by id |
| `POST` | `/uploads/presign`, `/uploads/complete` | Upload a CSV or model file straight to S3-compatible storage, then queue the job |
| `GET` | `/jobs/{job_id}` | Status and result of a background job |
| `POST` | `/analyze/model` | Scan a model file |
| `GET` | `/analyze/model/history`, `/analyze/model/{scan_id}` | Past scans |
| `GET` | `/trust/score` | Headline numbers of your latest analysis |
| `GET` | `/history`, `/history/{result_id}` | Your analyses and model scans (all of them, for an admin) |
| `DELETE` | `/history/{result_id}` | Delete a stored analysis |
| `GET`, `PUT` | `/results/{result_id}/marks` | Your trial marks ("real problem" or "fine") on the flagged items of your own result; a null verdict clears one |
| `GET` | `/marks/export` | All trial marks as CSV (admin) |
| `GET` | `/audit/events` | Audit log (your own events, or everyone's for an admin) |
| `POST` | `/reports/generate` | Build a report from the latest result (`?source=auto\|demo\|upload\|real_dataset`) |
| `GET` | `/forensics/latest`, `/forensics/narrative`, `/forensics/timeline`, `/blast-radius/latest` | Forensic views of the latest result |
| `GET` | `/defense/status`, `/defense/hitl/pending` | Defence actions and the human review queue |
| `POST` | `/defense/quarantine`, `/defense/hitl/decide` | Quarantine the latest flagged rows; decide a review case |
| `GET` | `/blueteam/status`, `/blueteam/incidents`, `/blueteam/playbooks`, `/blueteam/playbook/{attack_type}` | Response summary, incident log and playbooks |
| `GET` | `/users` | List users (admin) |
| `POST` | `/users` | Create a user (admin) |
| `DELETE` | `/users/{user_id}` | Delete a user (admin) |
| `POST` | `/demo/run` | Run the synthetic sample batch |
| `GET` | `/datasets/demo`, `/datasets/demo/samples` | The synthetic sample data |
| `POST` | `/detect/analyze` | Run the detector on sample rows |
| `GET` | `/detect/results/latest` | The latest detector result |
| `GET` | `/datasets/real` | List the reference datasets |
| `GET` | `/datasets/real/{name}/download` | Download one as CSV |
| `POST` | `/datasets/real/{name}/analyze` | Analyse one |
| `WS` | `/ws/v1/detection-stream` (no prefix) | Live analysis events, authenticated by the session cookie |

### Command line

`cli/veritas_cli.py` scans a model file or checks a dataset from a terminal or CI job, using an API key. It exits 1
when the verdict reaches `--fail-on`. See the docstring at the top of the file for usage.

---

## Sustainable Development Goals

Veritas was started for a Sustainable Development Goals hackathon. The connection is what it checks, not a promise
about outcomes:

| SDG | Connection |
|---|---|
| **SDG 3**, Good health and well-being | Medical and health datasets can be checked for poisoned rows and labels before a diagnostic model is trained on them. |
| **SDG 9**, Industry, innovation and infrastructure | Gives ML teams a measured data and model-file check to put in their training pipeline and CI. |
| **SDG 16**, Peace, justice and strong institutions | Gives institutions that train models on submitted data a documented, auditable check, with the false-alarm level and limits stated. |

---

## Licence

MIT. See [LICENSE](LICENSE).
