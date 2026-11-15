# Handover — Veritas detection engine (2026-11-08)

## Goal (set by Saikrishna, 2026-11-08)
Make Veritas a tool anyone can use on any dataset (tabular, images, audio, video), not just CSVs. Work on the
**detection engine first**; frontend, auth, deploy come after. Benchmark on **real downloaded datasets**, not
synthetically inflated ones. Target: find the poisoned samples/cells **at 1% poisoning**, reported honestly per
attack (some clean-label attacks may not be reachable at 1%; say so with numbers).

## State of the repo
- `main` is green in CI as of 2026-11-08 (backend sqlite+postgres, frontend, security, Playwright e2e, CodeQL). scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0.
- Product side (API, accounts, quotas, presigned uploads, CLI, account deletion) is built; deploy has never run (no `DEPLOY_*` secrets). Full list: "What is left" below.
- **Tabular detector** `veritas2` (`backend/app/detection/veritas2.py`): label check + gated shift check, frozen. Benchmark v2: `docs/benchmark-v2-findings.md`, `docs/model-card.md`.
- **Image track (new, this week)** in `backend/research/`:
  - `mm/datasets.py` real datasets (CIFAR-10/100, Fashion-MNIST, GTSRB via torchvision, MD5-verified), `mm/embed.py` ResNet-18 embeddings (GPU, cached), `mm/attacks.py` 7 pixel-level attacks, `mm/pixel.py` low-level pixel features, `mm/detector.py` `veritas_mm`, `bench_mm.py` runner, manifests `benchmark/manifest-mm-images-v{1,2,3}.json`.
  - Results in `backend/research/results/*benchmark-mm-images-*`. v1 = veritas2 only, v2 = + pixel check, v3 = pixel gate 3 -> 1 (chosen on dev), test split only.
  - `mm/gate_study_dev.py`: the dev-only study that picked the pixel gate.

## What the image results say (test = CIFAR-10 + GTSRB; dev = Fashion-MNIST + CIFAR-100)
- Label flips and backdoors that also change labels are found through the label check (AUPRC 0.4-0.65 at 1-5%).
- A ResNet embedding cannot see a corner patch or faint noise; the pixel-statistics check can. At 5% poison it takes patch backdoor 0.63 -> 0.95 and clean-label backdoor 0.04 -> 0.70.
- At 1%: pixel check alone ranks patch (0.71), clean-label (0.57), duplicates (0.79) well, but batch-level verdicts are weak for every detector (about 20% of attacked batches). Blended backdoor and class-noise attacks are still at or near chance at 1%.
- Clean-batch false alarms: dev 1.7% at the lowest cut-off (n=120); test was 12.5% in v2 (n=40, noisy). Needs more clean batches before trusting.
- Check `results/*images-v3*` for the held-out numbers of the gate-1 change (if the file is missing the run did not finish: `VERITAS_MM_MANIFEST=manifest-mm-images-v3.json PYTHONPATH=. python -m research.bench_mm --workers 6 --label v3`, about 25 min).

## Rules that keep the numbers honest
- Tune only on **dev** datasets; quote **test**. Never edit a manifest in place: new `benchmark_id` for any change.
- Attacks here are simplified published recipes written by us. A miss is a miss on the easy version.
- Real human label noise (CIFAR-10N) is reported separately (`bench_realnoise.py`), never mixed with injected-attack numbers.

## Setup
```bash
cd backend
uv venv ~/.cache/veritas-envs/mm --python 3.12
uv pip install --python ~/.cache/veritas-envs/mm/bin/python -r requirements.txt -r research/requirements.txt torch torchvision torchaudio
PYTHONPATH=. ~/.cache/veritas-envs/mm/bin/python -W ignore -m research.bench_mm --profile smoke   # about 2 min
```
- Datasets download to `~/.cache/veritas-bench` (override `VERITAS_BENCH_CACHE`), about 1.6 GB so far. Put the venv on a disk with several GB free (`/tmp` was too small).
- Full image benchmark: about 90 minutes on an RTX 2050 (4 GB). Run it detached (`setsid nohup ... & disown`); a session restart kills ordinary background jobs.

## Next steps, in order
1. Read the v3 test results; commit them with a short note in `docs/`.
2. 1% for the attacks still at chance: blended backdoor, class-noise. Ideas: spectral signatures / activation clustering on a model trained on the batch, frequency-domain trigger search, training-dynamics (per-sample loss) scores.
3. More clean batches (aim for 200+) to pin down the false-alarm rate; consider conformal calibration.
4. Cell-level localisation for tabular data (predict each cell from the others, flag outliers) so "which cells" is answered, not just "which rows".
5. Audio (Speech Commands, ESC-50) and video (UCF-101) tracks reusing the harness: raw data -> attack -> embed changed samples -> detectors. Add a per-modality attack list to `mm/attacks.py`.
6. Per-attack report (the product requirement): a score and plain verdict per attack family, not one combined verdict. Then the frontend report page.

## What is left outside detection
Needs accounts: deploy secrets (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`), SMTP, Turnstile, Sentry, real S3/R2 (presigned size enforcement only tested with moto), domain, hosting, uptime monitor. Needs a person: legal review of `docs/legal/*` drafts (placeholders). Can be built: staging stack (documented, never started), OAuth sign-in, docs site, frontend Terms/Privacy pages and browser presigned upload, Dependabot #2 nginx (test with `docker compose build`), #5 node 25 (skip), #16 pydantic-core (close), #14 tailwind 4 and #10 eslint 10 (frontend).

## Do not share
Your own tokens, `.env` files, or SSH keys. Your friend creates their own.
