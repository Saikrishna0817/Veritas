# Veritas — roadmap to a public launch

_Written 2026-11-06. Starting point: the audit fixes staged on `main` (99 tests
passing), benchmark v1 combined recall 0.0. All resources below are free or
have a free tier; check free-tier limits again before you rely on them, they change._

## 0. What "shippable" means here

Veritas is ready for the public internet when all five of these hold:

1. **It measurably works.** On a public, versioned benchmark built from real
   datasets, every verdict it shows beats a simple baseline, and the numbers
   are published on the site next to the verdicts.
2. **It is safe to let strangers use.** Self-serve sign-up, per-user data
   isolation, quotas, uploads in object storage, untrusted files never
   unpickled in the web process.
3. **It stays up.** A real job queue, Postgres, backups, monitoring and an
   on-call alert, on hosting that costs ~$0 at launch volume.
4. **It is honest.** No feature in the UI that is simulated (today: Layer 2
   activations, Layer 5 federation, the red-team "resilience score").
5. **It is legally deployable.** Terms, privacy policy, data-retention policy,
   delete-my-data button.

## 1. Scope decision (do this first, ~1 day)

Narrow the product to the two things that can be made real:

| Pillar | Promise | Why it is shippable |
|---|---|---|
| **Dataset audit** (tabular CSV/Parquet) | "Find mislabeled rows, suspicious clusters / trigger patterns, and drift against a trusted reference set." | Mature, free methods exist (confident learning, outlier detection, two-sample tests) and real-noise datasets exist to prove it. |
| **Model-file scan** (`.pkl`, `.joblib`, `.pt`, `.safetensors`, `.onnx`) | "Tell me whether this file can run code when loaded, and what's in it." | Deterministic problem; ground truth is easy to build; real user pain (malicious models on public hubs). |

**Cut or relabel:** Layer 2 (simulated activations), Layer 5 (simulated
federation), the "CONFIRMED_POISONED" verdict and A–F grade until the
benchmark supports them, and the red-team resilience score. Move them behind
a "Research preview" flag or delete them. Images/audio/LLM data stay out of
scope for v1.

## 2. Phase 1 — Benchmark v2 on real data (weeks 1–3)

The v1 benchmark is one synthetic batch per attack, so it can't tell a good
detector from a bad one. v2 must be large enough to give confidence intervals.

### 2.1 Datasets (all free)

**Clean tabular bases** (inject attacks with known ground truth):

| Source | What to take | Access |
|---|---|---|
| OpenML-CC18 suite | 72 curated classification datasets; pick ~15 spanning size 1k–500k rows, 5–500 features, binary + multi-class | `openml` Python package, CC-BY |
| UCI ML Repository | Adult, Bank Marketing, Covertype, HIGGS (subset), Credit default | `ucimlrepo` package |
| sklearn built-ins | Already used in `app/demo/real_datasets.py`; keep as fast CI smoke set | bundled |
| ADBench | 57 tabular anomaly-detection datasets with real anomaly labels | github.com/Minqi824/ADBench |

**Real (not injected) label noise** — the most important set, because it
tests on errors no one simulated:

| Source | Notes |
|---|---|
| CIFAR-10N / CIFAR-100N | Real human annotation noise on CIFAR; use precomputed embeddings (e.g. a free CLIP/ResNet feature extract) so it becomes a tabular task. github.com/UCSC-REAL/cifar-10-100n |
| Cleanlab label-error collection | Known label errors in popular test sets (labelerrors.com); ground truth from human re-verification |
| Food-101N / Animal-10N | Real web-label noise; also embedding-based. Free for research use — check the licence before using commercially |

**Poisoning / backdoor attacks** (attack implementations, not data):

| Source | Use |
|---|---|
| Adversarial Robustness Toolbox (ART, MIT) | Backdoor, clean-label, feature-collision poisoning generators, and reference defenses (Activation Defence, Spectral Signatures) to use as baselines |
| BackdoorBench (SCLBD) | Standardised backdoor attacks + defenses; reuse the attack definitions and reporting format |
| "Just How Toxic is Data Poisoning?" benchmark (aks2203/poisoning-benchmark) | Standard clean-label poison settings to report against |

**Model-file ground truth:**

| Source | Use |
|---|---|
| Self-generated benign models | Script that trains every sklearn/XGBoost/LightGBM/PyTorch type and saves in every format → negatives (extends `test_model_security.py`) |
| Self-generated malicious files | `__reduce__` payloads, nested pickles, joblib/zip tricks, `torch.load` with pickle inside, protocol 0–5 → positives. Generate locally; never download live malware |
| ProtectAI `modelscan`, Trail of Bits `fickling`, `picklescan` | Baselines to compare against, and their test corpora for edge cases |
| NIST TrojAI rounds (pages.nist.gov/trojai) | Free trojaned vs clean models; later, for a "backdoored model" detector |

### 2.2 Harness changes (`backend/research/`)

- New immutable `benchmark/manifest-v2.json`: dataset × attack × poison rate
  (1%, 3%, 5%, 10%) × 10 seeds, plus clean batches at the same count. That's
  ~3,000 scored batches for tabular audit.
- Report **row-level** AUROC/AUPRC and precision@k for row flags (what users act
  on), and **batch-level** TPR at a fixed 1% / 5% clean false-positive rate.
  Bootstrap 95% CIs for every number.
- Cache downloaded datasets under `~/.cache/veritas-bench` with SHA-256s in the
  manifest; CI runs the sklearn smoke subset, a weekly scheduled GitHub
  Action runs the full set (free on public repos).
- Output one JSON + one Markdown table per run into `research/results/`, and
  generate the public benchmark page from it.

### 2.3 Baselines (must exist before tuning anything)

| Task | Baselines |
|---|---|
| Label errors | cleanlab `find_label_issues` (out-of-fold probs from a GBM), kNN label disagreement |
| Outliers / triggers | IsolationForest, ECOD and kNN from PyOD, ART Spectral Signatures and Activation Clustering on model embeddings |
| Drift | Per-feature KS + Bonferroni, MMD with permutation p-value, classifier two-sample test (alibi-detect or own ~30 lines) |
| Model scan | modelscan, fickling, picklescan |

**Exit criterion:** benchmark v2 published in the repo, with baselines and
the current pipeline scored. Expect the current pipeline to lose; that's
what this phase is for.

## 3. Phase 2 — Rebuild the detector against the benchmark (weeks 3–7)

Rules: pre-register thresholds on a dev split of datasets, report on held-out
datasets only, and never tune on the test manifest.

1. **Label-issue detector** = confident learning on out-of-fold
   probabilities (5-fold, LightGBM default). This is the single highest-value
   signal; it should power the main "rows to review" list.
2. **Outlier / trigger detector** = ensemble of ECOD + IsolationForest +
   kNN distance, rank-averaged; plus spectral signatures computed on the
   GBM leaf embeddings for backdoor-style clusters.
3. **Drift vs reference** = classifier two-sample test with a permutation
   p-value, plus per-feature KS to explain *which* columns moved. Replaces
   L1's KL/JS heuristics.
4. **Keep L3/L4 only if they add lift** in ablation (benchmark with and
   without each). Delete what doesn't.
5. **Calibrated output:** replace the 0–1 "suspicion score" and A–F grade with
   (a) per-row ranked flags with reasons, (b) a batch p-value / "expected
   false alarms at this threshold", and (c) the measured precision on
   benchmark v2 at that threshold, shown in the UI.
6. **Reference split** becomes explicit: the user uploads a trusted reference
   file or picks a column/date cut-off (closes audit M3).
7. **Model scanner:** replace the custom allow-list unpickler as the *verdict*
   with static opcode analysis (fickling-style, never executes), keep the
   restricted loader only for metadata extraction, run in a sandboxed worker
   (separate container, no network, seccomp, CPU/mem limits). Add
   `safetensors`/ONNX support (safe formats, report "safe format").

**Exit criteria (public numbers, held-out datasets):**
- Label issues: AUPRC ≥ cleanlab baseline (being equal is fine, this *is* cleanlab
  plus explanations); precision@100 ≥ 0.6 on CIFAR-10N embeddings.
- Backdoor/outlier rows: beats best single PyOD baseline on AUROC at 5% poison.
- Drift: ≤ 5% false alarms on clean batches, ≥ 80% power at the documented shift size.
- Model scan: 100% of generated malicious files flagged, 0 false positives on
  benign corpus, ≥ parity with modelscan on its test set.

If a pillar misses its bar, ship it labelled "beta" with its real numbers, or don't ship it.

## 4. Phase 3 — Product hardening for strangers (weeks 5–9, overlaps Phase 2)

| Area | Change | Free resource |
|---|---|---|
| Database | SQLite → Postgres; add Alembic migrations back (properly this time); every table gets `owner_id` / `org_id` | Neon or Supabase free tier; local Postgres in compose |
| Tenant isolation | Every query scoped by owner; delete the global `upload_result_cache["latest"]`; per-user results/history (closes audit H7) | — |
| Files | Uploads to object storage with presigned URLs, size limits enforced at the edge, auto-delete after N days | Cloudflare R2 (free tier, no egress fees) |
| Jobs | Replace in-process `JobStore` with a real queue; workers in a separate container so a heavy scan can't stall the API | `arq` or RQ + Redis (Upstash free tier or Redis in compose) |
| Auth | Self-serve sign-up, email verification, password reset, optional GitHub/Google OAuth; move JWT from localStorage to httpOnly cookie + CSRF token | Resend or Brevo free tier for email |
| Abuse controls | Per-user quotas (rows/day, scans/day), Redis-backed rate limit (current one is per-process), Cloudflare Turnstile on sign-up | Cloudflare Turnstile (free) |
| API | API keys per user, versioned `/api/v1`, OpenAPI published; a small `veritas` CLI / GitHub Action that runs a scan in CI (good growth channel) | — |
| Frontend | Remove or flag simulated pages; results page = ranked rows + reasons + benchmark-backed precision; empty/error states; accessibility pass | — |
| Tests | Playwright e2e for login → upload → results → export; load test with k6 | both free |

## 5. Phase 4 — Deployment and operations (weeks 8–10)

**Recommended zero-cost stack** (check limits at sign-up):

| Need | Option |
|---|---|
| Compute (API + workers) | Oracle Cloud Always Free ARM VM (largest free CPU/RAM; runs the existing docker-compose). Fallback: Hugging Face Spaces (free CPU, but sleeps and is public) or Render free web service (sleeps) |
| DNS, TLS, CDN, WAF | Cloudflare free plan in front of the VM; Cloudflare Pages for the static frontend |
| Postgres | Neon / Supabase free tier, with daily `pg_dump` to R2 |
| Object storage | Cloudflare R2 |
| Errors | Sentry free tier (backend + frontend) |
| Metrics / logs | Grafana Cloud free tier (Prometheus metrics from FastAPI, Loki logs) |
| Uptime | UptimeRobot or Better Stack free monitors → email/Telegram alert |
| CI/CD | GitHub Actions: test → build images → push to GHCR → deploy over SSH on tag |
| Security scanning | Dependabot, CodeQL, `pip-audit`, Trivy on images (all free on public repos) |

Checklist: HTTPS only + HSTS, secrets in the host's env (never in images),
backups tested by an actual restore, a runbook in `docs/runbook.md`, a staging
environment (second compose project on the same VM).

## 6. Phase 5 — Launch (weeks 10–12)

- **Public pages:** landing page with the benchmark table, model card (updated
  `docs/model-card.md`), "how it works", limitations, changelog.
- **Legal:** Terms of Service, Privacy Policy (what's stored, retention,
  sub-processors: Cloudflare, Neon, Sentry…), delete-account-and-data flow,
  cookie notice only if you add non-essential cookies. Free generators exist,
  but have a human read them; GDPR applies once EU users sign up.
- **Docs site:** MkDocs Material on GitHub Pages.
- **Soft launch:** 10–20 invited users → fix → public post (Show HN, r/MachineLearning,
  MLOps communities) with the benchmark as the headline, not the UI.
- **Licence decision:** open-source the detector + benchmark (credibility,
  free CI minutes), keep the hosted service as the product.

## 7. Timeline summary (one developer, part-time ≈ 20 h/week)

| Weeks | Deliverable | Gate |
|---|---|---|
| 0 | Commit audit fixes; scope cut (§1) | Simulated features hidden |
| 1–3 | Benchmark v2 + baselines | Reproducible numbers in repo |
| 3–7 | Detector rebuild + sandboxed model scanner | §3 exit criteria met or pillar marked beta |
| 5–9 | Postgres, isolation, queue, auth, quotas | Two test users can't see each other's data (automated test) |
| 8–10 | Free-tier deployment, monitoring, backups | Restore drill passes; staging live |
| 10–12 | Legal, docs, soft launch | 10 external users complete a scan |

## 8. Risks

- **The detector may never beat the baselines.** Then the product is the
  baselines, well packaged, explained and benchmarked. That is still a real
  product (cleanlab + PyOD + drift tests + safe model scanning in one place,
  for people who don't write Python). Decide at the Phase 2 gate, don't slip.
- **Free tiers change or sleep.** Keep everything in docker-compose so moving
  to a ~$5/month VPS (Hetzner, etc.) is a one-hour job.
- **Hostile uploads.** The model scanner is the main attack surface; it must
  never deserialize in the API process (§3.7) before public launch.
- **Dataset licences.** Some noisy-label sets are research-only; use them for
  the public benchmark, but don't bundle them into the product.
