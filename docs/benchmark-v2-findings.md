# Benchmark v2 — baseline findings (2026-11-06)

Run `baseline-r1` of `veritas-tabular-poisoning-v2`: 2,520 batches on 18 real OpenML datasets.
Full tables: `backend/research/results/2026-11-06-benchmark-v2-baseline-r1.md` (JSON beside it).
Numbers below are the 12 held-out **test** datasets at 5% poison; nothing has been tuned on them yet.

**Caveat:** Veritas took ~37 s per batch (baselines 0.02–1.5 s), so it ran on 1 of 5 seeds
(`detector_seed_limits`): 12 test batches per cell instead of 60. Its confidence intervals are wide.

## Row level: AUPRC (random ranking = 0.05)

| attack | Veritas | best baseline |
|---|---|---|
| label flip (random) | 0.05 (chance) | cleanlab 0.67 |
| label flip (targeted) | 0.05 (chance) | cleanlab 0.73 |
| backdoor, out-of-range trigger | 0.38 | two-sample test 0.82 |
| backdoor, in-range trigger | 0.07 | two-sample test 0.64 |
| clean-label feature collision | 0.05 (chance) | two-sample test 0.86 |
| covariate shift | 0.07 | ECOD 0.28 |

## Batch level: share of attacked batches caught at 5% false alarms

Veritas: label flips 0–8%, backdoors 67%, collision 42%, shift 17%. Cleanlab: label flips 72–82%, backdoors ~64%.
Veritas's backdoor and collision numbers are on 12 batches each (CIs span ±25 points), so they are
not yet distinguishable from cleanlab's.

## What this says

1. **Veritas finds no label errors at all.** Its row detector never looks at labels, so label flips,
   the most common real-world data problem, score exactly at chance. cleanlab does this in ~1 s.
2. **Backdoors:** Veritas finds only out-of-range triggers (0.38); a plain classifier two-sample test
   finds both kinds far better (0.82 / 0.64).
3. **Speed is a product problem:** ~37 s per 3,000-row batch (up to ~130 s on 26-class data) is
   100–1,500× slower than the baselines it loses to.
4. **No attack type is best handled by any single method.** Label errors need a label-aware detector,
   backdoors and collisions need a two-sample/classifier test, shift needs a feature-level test.
   That matches the Phase 2 plan: combine cleanlab + a classifier two-sample test + outlier scores.

## Known benchmark weaknesses (fix before quoting publicly)

- The two-sample test (`c2st`) is strong on rows but weak on batch-level TPR; its batch statistic
  (AUC vs. clean batches) needs a permutation p-value instead.
- Dev and test datasets were chosen by size/diversity, not stratified; 12 datasets give wide CIs.
- (Resolved in the Phase 2 update below) real-noise track added; the old pipeline stays on 1 seed because of its runtime.
- Injected poisoning is still synthetic; only label noise has a real-data check.

---

# Phase 2 update (2026-11-07): `veritas2` and the real-noise track

## What `veritas2` is

`backend/research/baselines.py::veritas2` (frozen after dev-only tuning; see the comment there):

1. **Label check** (always on): gradient-boosted classifier trained on clean reference rows;
   incoming rows are ranked by cleanlab's label-quality score.
2. **Shift check** (gated): classifier two-sample test, "can a model tell incoming rows from
   held-out clean rows?". It joins the row ranking only when its own batch-level tail test
   reaches -log10 p >= 4, so it cannot dilute label rankings when nothing has shifted.
3. Evidence is converted to p-values against held-out clean rows; batch score is the strongest tail test.

Tuning used the 6 dev datasets only. What dev showed: label+shift beat label alone on backdoors and
collisions; adding ECOD/IsolationForest hurt (their noise lifted the clean-batch threshold); a plain
max over components diluted label rankings, which the gate fixes. Gate 3/4/6 tied; 4 was kept.

## Held-out test datasets (12 datasets, injected attacks, 5% poison)

| | old Veritas | **veritas2** | cleanlab | two-sample test |
|---|---|---|---|---|
| mean row AUPRC (random = 0.05) | 0.11 | **0.69** | 0.54 | 0.49 |
| mean batch TPR at 5% FPR | 0.33 | **0.78** | 0.55 | 0.14 |
| sec / batch | 37 | 1.8 | 0.8 | 1.5 |

Per attack, `veritas2` is at or above the best single method everywhere except random label flips
(0.66 vs cleanlab 0.67, inside the CI). Weakest: covariate shift (AUPRC 0.40 [0.24, 0.57]).
Full tables: `research/results/2026-11-06-benchmark-v2-final-r1.md` (old Veritas numbers are from
`baseline-r1`, 1 seed).

## Real human label noise (CIFAR-10N, nothing injected)

3,000-image batches, 20 seeds. Row AUPRC where random = the mistake rate (5% column shown):

| noise source (wrong labels) | old Veritas | veritas2 | cleanlab |
|---|---|---|---|
| aggregate (9%) | 0.05 | 0.45 [0.43, 0.47] | 0.46 [0.44, 0.48] |
| random1 (17%) | 0.05 | 0.54 [0.49, 0.58] | 0.56 [0.53, 0.59] |
| worse (40%) | 0.05 | 0.57 [0.53, 0.61] | 0.58 [0.55, 0.61] |

## Honest reading

- **The old pipeline is at chance on real label noise** and takes ~130 s per batch. Replacing it is justified.
- **On real human mistakes `veritas2` equals cleanlab, it does not beat it** (it trails by 0.01-0.02, mostly inside the CIs). The gain over cleanlab is on injected backdoors/collisions/shift, not on label noise.
- **Batch-level detection of small label-noise rates is weak for everyone.** At a 3-5% mistake rate, whole-batch TPR at 5% FPR is 0.05-0.8 with very wide CIs; at 10% it is 0.95-1.0. `veritas2` is behind cleanlab at 5% on the aggregate labels (0.25 vs 0.75, CIs overlap widely). Likely cause: the shift test adds noise to the clean-batch threshold. Not yet fixed; do not tune it on this track.
- The ranking is useful ("review these rows first"); the batch verdict is not yet reliable for small contamination.
- The real-noise data is one dataset family (CIFAR-10 embeddings); old Veritas ran on 5 of 20 seeds.

## Still open

- ~~Move `veritas2` into the app pipeline~~ done (see Phase 2b below).
- ~~Explicit reference-set input~~ done: separate reference file, column cutoff (numbers/dates), or file order (default, with a warning).
- ~~Fix batch-level power at small contamination~~ partly done (Phase 3 below): calibration fixed, power at 1-3% is a sample-size limit.
- ~~Sandboxed model-file scanner~~ built (Phase 3c), not yet benchmarked at scale; more real-noise datasets (CIFAR-100N, text embeddings).

---

# Phase 2b (2026-11-07): `veritas2` is now the app's default engine

`DetectionPipeline(engine="v2")` (the default) runs `app/detection/veritas2.py`; `engine="legacy"` keeps the
original five layers for comparison. The benchmark imports the shipped module, and re-running four real-noise
batches reproduced the stored scores exactly. End to end on a synthetic CSV upload (3,000 rows, 3 classes):
clean file -> CLEAN, 10% flipped labels -> CONFIRMED_POISONED, 58 rows flagged of which 55 were the flipped ones, 1.3 s.

Verdict cut-offs are batch evidence (-log10 p) read off held-out clean batches: LOW_RISK >= 4.0, SUSPICIOUS >= 5.5
(~5% of clean batches reach it), CONFIRMED >= 9.5 (~1%). They map onto the app's 0.15 / 0.35 / 0.65 suspicion
thresholds. **"CONFIRMED_POISONED" now means "evidence at the 1% false-alarm level", not a causal proof**: the
causal layer no longer runs. Rows are flagged only when the batch itself is at least SUSPICIOUS.

Not changed: the model-file scan still feeds extracted parameters through the same pipeline, and has no benchmark.

# What was wrong with the old pipeline (diagnostic run)

24 batches (3 dev datasets, 1 seed, 10% poison, 2 clean batches per dataset), every layer's score recorded.
Exploratory, not a benchmark: it is small and nothing here was used for tuning. Script output is not committed.

1. **The blend can never reach a verdict.** Layer 3 flagged *every* out-of-range backdoor row (31/31 and 55/55) and
   almost nothing else, yet the batch verdict was LOW_RISK (overall 0.21-0.34), because one layer firing is weighted
   at 0.20 and the rest sit near zero. Across all 18 attacked batches the overall score averaged 0.11 against 0.03-0.05
   on clean batches, and no attacked batch reached SUSPICIOUS (0.35); the highest was 0.34. A weighted average needs several
   layers to agree; these attacks trip one at a time.
2. **No layer looks at labels properly.** Label flips (random and targeted) scored 0.03-0.15, equal to clean batches.
   Layer 3 flagged 0-4 of 23-55 flipped rows, which is chance. Only the label-ratio shift in Layer 1 reacts, and only
   to targeted flips in 2 of 3 datasets.
3. **Layer 4 (causal) never fired and costs the most time.** `proof_valid` was false in all 24 batches and the causal
   effect was ~0.01 everywhere (including 10% poisoning). It took 2-4 s per batch on sets under 1,000 rows (about 60%
   of the run on qsar-biodeg), and it retrains models, so it grows with data size. Since `CONFIRMED_POISONED`
   required it, the top verdict was unreachable.
4. **Layer 2 adds a constant.** Spectral score is a fixed 0.10-0.20 on clean and attacked batches alike (it is 20% of
   the weight), which only lifts every batch's floor.
5. **Layer 1's KL and Mahalanobis alarms never fired** on any of the 24 batches; only the blend-trigger heuristic did,
   and it fired on in-range backdoors and targeted flips inconsistently (0.32-0.37 vs 0.02 clean).
6. **Layer 3 is a decent out-of-range row detector** (31/31 and 55/55 on `backdoor_ood`), but it is label-blind and
   missed in-range backdoors (1 of 31 and 1 of 55) and feature collisions (0). That is the one piece worth keeping
   in spirit; the shift check in `veritas2` does the same job with a trained discriminator and also finds in-range cases.
7. **Speed:** timing was only measured per layer on small datasets (wdbc 0.8 s, qsar-biodeg 4.5 s); the 37 s average in
   the full benchmark comes from larger sets and was not broken down per layer.

Net: the old design (hand-weighted average of five mostly label-blind layers, with the top verdict gated on a proof that
never validates) was not recoverable by re-weighting, which matches the earlier benchmark-v1 finding of 0.00 F1 per layer.
The legacy code is retained only so the benchmark can keep comparing against it; decide later whether to delete layers 1-5.

---

# Phase 3 (2026-11-07): batch-level power

**Cause found:** the batch statistic was miscalibrated, not weak. The binomial tail test treats the clean 99% quantile as
exact, but it is estimated from ~540 calibration rows (the ~5 largest), so its noise leaks into the p-value. Clean batches
reached -log10 p = 4.6 at the 95th percentile (it should be 1.3), which made the empirical threshold high for every attack.

**Tried on the 6 dev datasets only** (batch TPR at 5% false alarms, mean over attack types):

| variant | 1% | 3% | 5% | 10% |
|---|---|---|---|---|
| veritas2 as frozen (binomial tail) | 0.18 | 0.32 | 0.67 | 0.81 |
| + Mann-Whitney mean test | 0.18 | 0.32 | 0.67 | 0.81 (no change) |
| out-of-fold labels on the whole reference (3x clean rows) | 0.13 | 0.27 | 0.64 | - (worse) |
| exact hypergeometric tail, label + shift | 0.16 | 0.41 | 0.63 | 0.79 |
| same, shift weighted by 4:1 (**chosen**, penalty log10 4) | 0.23 | 0.46 | 0.68 | 0.83 |
| shift penalties 1.0 / 1.5 | 0.29 / 0.33 | 0.42 / 0.40 | 0.65 / 0.60 | 0.82 / 0.80 |

The exact test (`exact_tail_test`) conditions on the pooled order being random when the batch is clean, so it is
valid at any calibration size; clean batches now reach 1.3 at the 95th percentile. The 4:1 weight is a weighted
Bonferroni split of the false-alarm budget between the two checks (label 80%, shift 20%); the heavier penalties
traded away collisions and backdoors for label flips, so 4:1 was kept as the middle.

**Held-out result (12 test datasets; real-noise track, which was not used for tuning):**

| batch TPR at 5% false alarms | old veritas2 | new veritas2 | cleanlab |
|---|---|---|---|
| test, 1% poison | 0.11 | 0.15 | 0.14 |
| test, 3% | 0.65 | 0.65 | 0.49 |
| test, 5% | 0.78 | 0.75 | 0.55 |
| test, 10% | 0.89 | 0.89 | 0.64 |
| CIFAR-10N, 3% mistakes (mean of 3 sources) | 0.25 | 0.37 | 0.38 |
| CIFAR-10N, 5% | 0.53 | 0.72 | 0.78 |
| CIFAR-10N, 5%, aggre labels only | 0.25 | 0.55 [0.30, 1.00] | 0.75 [0.15, 0.95] |
| CIFAR-10N, 10% | 0.97 | 0.97 | 0.98 |

Row-level AUPRC is unchanged (the row ranking was not touched).

**Honest reading:**
- On tabular data nothing improves at the 5%-false-alarm operating point (differences are within noise). On real human
  noise the gap to cleanlab closes (aggre 5%: 0.25 -> 0.55; cleanlab 0.75, CIs overlap widely).
- The real gain is **calibration**: the batch score is now a proper p-value. The shipped verdict cut-offs are p <= 0.05
  (LOW_RISK), 0.01 (SUSPICIOUS), 1e-4 (CONFIRMED). On held-out clean batches: 7.1% reached 1.3 (nominal 5%, n=240),
  0.4% reached 2.0, none reached 4.0; on CIFAR clean batches 2.5% / 0% (n=40). At SUSPICIOUS (2.0) a 5%-poisoned batch is
  caught 68% of the time with ~0.4% false alarms; before, the SUSPICIOUS cut-off sat at about 5% false alarms for 75%.
- **Small rates stay hard.** At 1% poison (about 12 rows in 1,200) every method catches ~15%; at 3% about half. That is a
  sample-size limit of any single-batch test, not something more tuning fixes. The honest product statement is "reliable
  at >= 5% contamination; for less, use the ranked row list".
- Old `veritas2` binomial results remain in `results/*final-r1*`; `v2_ls_gate4` in `baselines.py` is the same detector.
- CIs from 40 clean batches (test: 240) give only a rough read on the false-alarm rates above.

# Phase 3b: legacy removal and explicit reference set

The five-layer code, SHAP drift, the v1 harness (`research/evaluate.py`) and the synthetic federated page were deleted;
tag `legacy-five-layer-pipeline` has them. Roadmap item 6 (audit M3) is done: the upload takes a trusted
`reference_file`, or `split_column` + `split_cutoff`, else the first 70% in file order, and the UI has a selector.

Simulated features removed: the sophistication score, counterfactual, red-team simulator with its "resilience score"
(routes, table writes, blue-team resilience panel, admin panel), and the fabricated timeline/schedule/source fields in the
injection narrative (ingestion time is not injection time). The narrative now states the flagged-row count, the two check
results and the average feature difference, and says it cannot say who or when. The attack-type guess is limited to the
three types the checks can support (label flip, backdoor, clean label). The blast radius reports flagged-row counts only.

# Phase 3c: model-file scanner (roadmap item 7)

`app/scanner/model_scan.py` (standard library only) answers "can loading this file run code?" without ever loading it.
- **Pickles** (pickle, joblib, legacy and zip PyTorch, `.npy` object arrays, gzip/bz2/xz wrapped): every opcode is walked with
  `pickletools.genops`; each import (GLOBAL, STACK_GLOBAL with memo tracking, INST, extension codes) is classed dangerous
  (os, subprocess, builtins.eval/exec/getattr..., socket, importlib, ctypes, ...), known-safe (numpy/sklearn/scipy/torch data
  containers) or unknown. Unresolvable imports count as dangerous, unknown ones as REVIEW. Payloads appended after a clean
  pickle are scanned too.
- **safetensors / GGUF / plain `.npy`**: structure validated (header size, offsets in range, no overlaps).
- **ONNX**: wire-format parse for custom operator domains (native code) and external-data paths that escape the folder.
- **Zip**: pickles inside are scanned; executable members, path-traversal names and zip bombs are flagged.
- **HDF5/Keras `.h5`** and unknown formats are reported as not inspected (REVIEW), never as safe.
- Verdicts: DANGEROUS, REVIEW, NO_CODE_EXECUTION_FOUND, SAFE_FORMAT. The scan runs in a separate `python -I` process with an empty
  environment, a throwaway cwd and memory, CPU, file-descriptor and write limits; a crash or timeout is itself DANGEROUS.
  Network isolation and seccomp are left to the deployment (run it in a no-network container).
- The sklearn restricted loader only runs afterwards, for metadata. Parameter-level "poisoning" analysis of weights was removed.

**Verified so far:** 29 unit tests with generated fixtures (reduce gadgets at pickle protocols 0/2/4/5, appended payload, gzip,
decompression bomb, torch zip and legacy torch, sklearn model, safetensors, ONNX, npy, HDF5, sandbox failure). That is not
the roadmap's exit bar: it has not been run against a public corpus of malicious models or a benign model-hub sample, and
has not been compared with modelscan/picklescan. Known gaps: HDF5/Keras, TensorFlow SavedModel, nested archives, pickles that
build their call chain without any dangerous import (the unknown-import rule covers custom classes only).

# Phase 4: scanner benchmark and first hardening slice

- **Scanner benchmark:** see `docs/scanner-benchmark.md`. It found a real hole (dotted attribute paths through allowed libraries,
  0/42 files flagged at all on the first held-out run) and benign joblib files that the first parser rejected; both are fixed and covered by tests.
- **Tenant isolation (roadmap Phase 3):** results, model scans, defense actions and review cases carry `owner_id`; every read takes an
  explicit owner (`ALL_OWNERS` for admins), so a forgotten filter is a `TypeError`. The global `latest` caches, live websocket events,
  history, reports, forensics, trust score, review queue and job lookups are per user. `tests/test_tenant_isolation.py` runs two users
  through all of that; the roadmap gate "two test users can't see each other's data" now has an automated test.
- **Quotas and data control:** daily per-user analyses, rows and scan-MB limits (admins exempt, `VERITAS_QUOTA_*`), `GET /account/usage`,
  `DELETE /account/data` (delete-my-data) and a startup retention purge (`VERITAS_RETENTION_DAYS`, default 30).
- **Still open in Phase 3:** Postgres + migrations (SQLite still), object storage for uploads, a real job queue and separate workers,
  self-serve sign-up/email verification/password reset, JWT in an httpOnly cookie with CSRF, Redis-backed rate limit (the limiter and
  the quotas' atomicity are single-process/SQLite), API keys and CLI, Playwright e2e and load tests. Pre-existing legacy rows with no owner
  are visible only to admins.

# Phase 5: hardening completed and verified (2026-11-07)

Verified by running it, not just writing it:
- **Both databases:** the full backend suite (161 tests) passes on SQLite and on PostgreSQL 16 (`VERITAS_TEST_PG_URL`), including a real RQ
  worker process fed through Redis (`VERITAS_TEST_REDIS_URL`) and the Redis-shared rate limiter. The first PostgreSQL run caught a dialect bug
  and the first live-server run caught a startup bug (`get_stats` after the owner change); both fixed.
- **Real stack:** `docker compose up --build` (Postgres, Redis, API, worker, nginx). Through nginx: cookie sign-in, CSRF enforced (403 without
  the header), background upload -> Redis -> worker -> Postgres -> `CONFIRMED_POISONED` in about 2 s.
- **Browser:** Playwright (3 tests, Chromium via the installed Brave) signs in, checks the session cookie is httpOnly and localStorage holds no
  token, uploads a CSV through the UI and sees the verdict and "Label Check"; anonymous users are redirected and get 401; sign-up shows the next step.
- **Load (k6, 5 virtual users, 30 s, one laptop):** history reads p95 13 ms, enqueue p95 25 ms, no errors other than the intended 429s.
  This is a smoke test, not capacity planning.
- **Backup:** `scripts/backup.sh` + `scripts/restore-drill.sh` restored a real dump into a scratch database and matched row counts.
- **Dependencies:** `pip-audit` found 16 known issues in the pins (starlette, python-multipart, pyasn1, anyio, certifi, idna, python-dotenv,
  python-jose with no fix); upgraded, `python-jose` replaced by PyJWT, now clean; `npm audit --omit=dev` clean; frontend lint and build pass.

Built in this phase: self-serve sign-up with email verification, password reset (single-use hashed tokens, no account enumeration), Turnstile
hook, API keys (hashed), httpOnly cookie sessions with double-submit CSRF and a WebSocket origin check, durable jobs table + blob storage
(local or S3/R2) + separate worker, Redis rate limits, `veritas` CLI and GitHub Action, Prometheus `/metrics` (token-gated) and Sentry hook,
compose stack, CI (tests on both databases, lint/build, pip-audit, Trivy, image push, SSH deploy), CodeQL, Dependabot, runbook (`docs/runbook.md`).

Not verified / not done:
- CI workflows, the deploy job, CodeQL and Dependabot have never run (they need the repository on GitHub with secrets).
- SMTP delivery, Turnstile, Sentry, S3/R2 uploads against a real service (S3 is tested against `moto` only), Cloudflare, Oracle Cloud, Neon/Supabase, uptime monitors: all need your accounts.
- Direct-to-storage presigned uploads were not built; uploads still pass through the API (limit 200 MB, nginx 210 MB).
- OAuth (GitHub/Google) sign-in, accessibility pass, a published OpenAPI/CLI package, a staging environment, and legal pages (Terms, Privacy) are not done.
- The Playwright suite is not in CI yet; the load test is not scheduled.
- One API process: WebSocket events and a few caches are per-process; result pages must poll jobs when workers run elsewhere.

## Dependency bump check (numpy 2.5.4, pandas 3.1.0)

Re-ran the 12 held-out test datasets (1,680 batches) after the Dependabot merges. All 12,000 per-batch detector outputs are identical to `final-r2` (bit-for-bit apart from timings), so the bumps did not move any score. `fetch_openml`'s default ARFF parser breaks on categorical columns under pandas 3.1, so the harness now falls back to `parser="liac-arff"`. Result: `backend/research/results/2026-11-07-benchmark-v2-deps-bump-numpy25-pandas31.*`.

## Dependency bump check (scikit-learn 1.9.1)

Re-ran the same 1,680 held-out batches on scikit-learn 1.9.1. Unlike the numpy/pandas bump, outputs are **not** bit-identical: about 45% of per-batch scores move (gradient-boosting internals changed), by up to 4.3 in `veritas2` batch score on individual batches. Aggregates stay within noise. `veritas2` mean batch detection at 5% false alarms by poison rate (1.8.0 → 1.9.1): 1% 0.150 → 0.144, 3% 0.647 → 0.611, 5% 0.747 → 0.744, 10% 0.886 → 0.872; mean row AUPRC at 5% poison 0.678 → 0.681 (95% intervals are about ±0.15 wide). No claim in the model card changes. The metadata unpickler allow-list also needed `sklearn._loss._loss` classes for 1.9 pickles. Result: `backend/research/results/2026-11-08-benchmark-v2-deps-bump-sklearn19.*`.

## Cell-level localisation (2026-11-13)

`veritas2.detect` now also returns `cell_scores`, shape (incoming rows, features + 1), the last column being the label:
the row score plus how unusual the cell is. A feature value scores the -log10 share of clean values at least as far from
the column mean, less log10(number of features); the label cell scores the label check. Ground truth is every value and
label the attack changed (`bench_v2.make_batch`).

Chosen on dev (`research/tab/cell_study_dev.py`) over: ridge residuals from the other features, their max with the
marginal score, min(row, cell), and no column-count correction. Uncorrected marginal scores were slightly better on dev
at 1% (0.28 vs 0.25) but blamed feature columns for label flips; the corrected version wins at 5-10% and inside poisoned rows.

Held-out test (12 datasets, 1,440 attacked batches), cell AUPRC; random ranking = the share of changed cells (0.1-0.7%
at 1% poison). "Row only" gives every cell of a row its row score.

| Attack | 1% cells / row only | 5% cells / row only | In-row column ranking |
|---|---|---|---|
| Random label flip | 0.47 / 0.04 | 0.64 / 0.04 | 0.99 |
| Targeted label flip | 0.55 / 0.06 | 0.71 / 0.05 | 0.99 |
| Out-of-range backdoor | 0.39 / 0.08 | 0.81 / 0.22 | 0.98 |
| In-range backdoor | 0.19 / 0.10 | 0.47 / 0.19 | 0.84 |
| Feature collision | 0.08 / 0.20 | 0.30 / 0.53 | 0.74 |
| Covariate shift | 0.01 / 0.01 | 0.15 / 0.10 | 0.56 |
| Mean | 0.28 / 0.08 | 0.51 / 0.19 | 0.85 |

Full table (3% and 10% too): `backend/research/results/2026-11-13-tabular-cells-test.md`.

- Inside a truly poisoned row the changed cells are ranked first almost always for label flips and out-of-range backdoors.
- Feature collision changes every feature of a row towards a victim, with in-range values; giving the whole row its row score
  is better (0.20 vs 0.08 at 1%). Covariate shift (+1 std on 30% of columns) is barely found at any level, because the row
  ranking itself is weak there.
- Cell quality is capped by the tabular row ranking, which has not had the image track's upgrades (full reference set,
  cross-fitted label check, several tail sizes). Those are the next lever.
- Not yet shown in the app: `pipeline.py` passes only row scores to the report.

## Row checks upgraded with the image track's lessons (2026-11-13)

`veritas2` now: label check cross-fitted over all reference rows (5 GBM folds; one fold model per incoming row for the
batch test, the mean of folds for the row list); shift check against all reference rows; batch evidence = best exact
tail test over the top 0.03/0.1/0.3/1%, minus log10(4); penalties label 0.1, shift 1.0 (gate 3 unchanged); row scores
extrapolated beyond the top clean row; cell scores compare with all reference rows.

Chosen on dev (`research/tab/row_study_dev.py`, 720 attacked + 360 clean batches). Rejected there: logistic regression
for the label check (faster, slightly worse rows), a per-claimed-class label test (no gain), and a marginal-outlier
row check (38% of clean batches flagged: in-sample reference scores are not comparable to incoming ones).

Held-out test (12 datasets, 1,440 attacked + 240 clean batches; `results/2026-11-13-benchmark-v2-final-r3-crossfit.*`;
"before" is the sklearn 1.9 run of the previous detector on the same batches):

| | Before | Now | cleanlab |
|---|---|---|---|
| Row AUPRC at 1% / 3% / 5% / 10% | 0.36 / 0.57 / 0.68 / 0.80 | 0.43 / 0.70 / 0.78 / 0.88 | 0.36 / 0.49 / 0.54 / 0.60 |
| Batch detection at 5% clean false alarms, 1% / 3% / 5% / 10% | 0.14 / 0.61 / 0.74 / 0.87 | 0.44 / 0.76 / 0.84 / 0.94 | 0.14 / 0.47 / 0.55 / 0.63 |
| Flagged at LOW_RISK (1.3), 1% / 5% | 19% / 76% | 37% / 82% | 51% / 72% |
| Clean flagged at 1.3 / 2.0 / 4.0 (nominal 5 / 1 / 0.01%) | 7.1 / 0.8 / 0% | 3.3 / 1.3 / 0% | 27.5 / 18.3 / 6.2% |
| Cell AUPRC at 1% / 5% | 0.28 / 0.51 | 0.35 / 0.60 | - |
| Seconds per batch | 1.6 | 5.4 | 0.7 |

Per attack at 1% (row AUPRC / flagged at 1.3): random flip 0.51 / 42%, targeted flip 0.60 / 37%, out-of-range backdoor
0.52 / 55%, in-range backdoor 0.49 / 37%, feature collision 0.39 / 43%, covariate shift 0.04 / 8%.

- The calibrated batch detection at 1% tripled (0.14 -> 0.44); every rate and every attack improved.
- Clean false alarms are now under nominal at 1.3 (was over); 1.3% at 2.0 is over nominal by one batch (3 of 240).
- cleanlab's raw cut-offs are not calibrated (27.5% of clean batches at 1.3), so its "flagged" share is not comparable.
- Covariate shift is still barely found at 1%; it needs a check built for many small column shifts.
- `bench_v2._openml` now falls back to the liac-arff parser on any parse error: inside worker processes the pandas 3.1
  failure arrives wrapped by sklearn's retry helper, not as a TypeError, and killed the first run of this benchmark.
