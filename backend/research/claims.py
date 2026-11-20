"""Every headline number in docs/model-card.md, the run that produced it, and the interval a rerun must land in.

Run from ``backend/``:
    python -m research.claims                  # recompute each claim from the committed result files (seconds)
    python -m research.claims --fresh          # check the reruns `make reproduce` wrote (label repro-*) instead
    python -m research.claims --rerun [--only images video]   # rerun the benchmarks (about a day in all), then --fresh
    python -m research.claims --commands       # print the command that reruns each benchmark
    python -m research.claims --table          # the claims table in the model card

A claim's ``text`` must appear in the model card, so the card and this file cannot drift apart silently.
Intervals: row AUPRC +-0.03, shares of attacked batches or runs +-0.05, clean-batch alarm shares +-0.02 (one clean
batch of 60 is 0.017). A rerun outside its
interval means the detector, a dependency or the data changed; say which before changing a claim, and give the new
numbers a new result file rather than editing an old one.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CARD = ROOT.parents[1] / "docs" / "model-card.md"
LOW_RISK, SUSPICIOUS = 1.3, 2.0            # evidence cut-offs (veritas_mm and veritas2 share them)

# name: (result file stem, environment, module and arguments). Reruns write <date>-<kind>-repro-<label>.
RUNS = {
    "tables":        ("2026-11-13-benchmark-v2-final-r4-resid-attack", "VERITAS_TAB_MANIFEST=manifest-v2.json", "bench_v2 --splits test"),
    "tables-breadth": ("2026-11-18-benchmark-v2-breadth-v1", "VERITAS_TAB_MANIFEST=manifest-v2-breadth-v1.json", "bench_v2 --splits test"),
    "images":        ("2026-11-16-benchmark-mm-images-v9-sig", "VERITAS_MM_MANIFEST=manifest-mm-images-v9.json", "bench_mm --splits test"),
    "images-bb":     ("2026-11-16-benchmark-mm-images-ext-v1", "VERITAS_MM_MANIFEST=manifest-mm-images-ext-v1.json", "bench_mm --splits test"),
    "images-breadth": ("2026-11-16-benchmark-mm-images-breadth-v1", "VERITAS_MM_MANIFEST=manifest-mm-images-breadth-v1.json", "bench_mm --splits test"),
    "label-errors":  ("2026-11-16-labelerrors-v1", "", "bench_labelerrors"),
    "adaptive":      ("2026-11-18-benchmark-adaptive-images-v2", "VERITAS_ADAPTIVE_MANIFEST=manifest-adaptive-images-v2.json", "bench_adaptive"),
    "audio":         ("2026-11-16-benchmark-audio-test-v3-sig", "VERITAS_AUDIO_MANIFEST=manifest-audio-v3.json", "bench_audio --splits test"),
    "audio-breadth": ("2026-11-16-benchmark-audio-breadth-v1", "VERITAS_AUDIO_MANIFEST=manifest-audio-breadth-v1.json", "bench_audio --splits test"),
    "video":         ("2026-11-14-benchmark-video-test-v2", "VERITAS_VIDEO_MANIFEST=manifest-video-v2.json", "bench_video --splits test"),
}


def _batches(r, rate=None, attack=None, data=None, clean=False):
    return [b for b in r["batches"] if b["split"] == "test" and (data is None or b["data_id"] == data)
            and ((b["attack"] is None) if clean else (b["attack"] is not None and (rate is None or b["rate"] == rate)))
            and (attack is None or b["attack"] == attack)]


def auprc(r, det, rate, data=None):
    """Mean row AUPRC over attacked test batches at this poison rate."""
    return statistics.mean(b["detectors"][det]["auprc"] for b in _batches(r, rate, data=data))


def flagged(r, det, rate=None, at=LOW_RISK, data=None, attack=None, clean=False, per_attack=None):
    """Share of test batches with evidence >= at; per_attack='min'/'max' takes that over the attacks."""
    def share(bs):
        return sum(b["detectors"][det]["batch_score"] >= at for b in bs) / len(bs)
    if per_attack:
        attacks = {b["attack"] for b in _batches(r, rate, data=data)}
        return {"min": min, "max": max}[per_attack](share(_batches(r, rate, a, data)) for a in attacks)
    return share(_batches(r, rate, attack, data, clean))


def tpr(r, det, rate):
    """Batch detection rate at 5% clean false alarms, mean over attacks (the harness's cluster-bootstrap point)."""
    s = r["summary"]["test"][det]
    return statistics.mean(v[str(rate)]["batch_tpr"][0] for k, v in s.items() if not k.startswith("_"))


def label_errors(r, data, stat, scorer="veritas_mm", mode="A_with_reference"):
    v = r["datasets"][data]["modes"][mode]["scorers"][scorer]["confirmed"][stat]
    return v[0] if isinstance(v, list) else v


def adaptive(r, stat, family=None, rate=None, s=None, min_asr=None):
    """Over the evaluated runs: share flagged (evidence >= 1.3) or mean attack success."""
    runs = [x for x in r["runs"] if x["kind"] == "eval" and (family is None or x["family"] == family)
            and (rate is None or x["rate"] == rate) and (s is None or x["s"] == s)
            and (min_asr is None or x["asr"] >= min_asr)]
    return statistics.mean((x["evidence"] >= LOW_RISK) if stat == "flagged" else x["asr"] for x in runs)


A, S, C = 0.03, 0.05, 0.02   # interval half-widths: AUPRC-type, share of attacked batches or runs, clean-alarm share
# (run, text in the model card, function, kwargs, expected, half-width)
CLAIMS = [
    ("tables", "| row AUPRC at 5% poison (random = 0.05) | 0.11 | 0.69 | **0.80**", auprc, dict(det="veritas2", rate=0.05), 0.80, A),
    ("tables", "| batch detection rate at 5% false alarms | 0.33 | 0.78 | **0.86**", tpr, dict(det="veritas2", rate=0.05), 0.86, S),
    ("tables", "Nov 13: 1% 0.47", tpr, dict(det="veritas2", rate=0.01), 0.47, S),
    ("tables", "3% 0.79", tpr, dict(det="veritas2", rate=0.03), 0.79, S),
    ("tables", "10% 0.94", tpr, dict(det="veritas2", rate=0.1), 0.94, S),
    ("tables", "On held-out clean batches 3.7% reached LOW_RISK", flagged, dict(det="veritas2", clean=True), 0.037, C),
    ("tables", "1.3% reached SUSPICIOUS", flagged, dict(det="veritas2", clean=True, at=SUSPICIOUS), 0.013, C),
    ("tables-breadth", "row AUPRC 0.71 at 1%", auprc, dict(det="veritas2", rate=0.01), 0.71, A),
    ("tables-breadth", "and 0.92 at 5% (0.73 to 0.99 by dataset)", auprc, dict(det="veritas2", rate=0.05), 0.92, A),
    ("tables-breadth", "64% / 95% of attacked batches flagged", flagged, dict(det="veritas2", rate=0.01), 0.64, S),
    ("tables-breadth", "64% / 95% of attacked batches flagged", flagged, dict(det="veritas2", rate=0.05), 0.95, S),
    ("tables-breadth", "1.9% at LOW_RISK", flagged, dict(det="veritas2", clean=True), 0.019, C),
    ("images", "row AUPRC 0.87 at 1% poison", auprc, dict(det="veritas_mm", rate=0.01), 0.87, A),
    ("images", "and 0.92 at\n5%", auprc, dict(det="veritas_mm", rate=0.05), 0.92, A),
    ("images", "83% of attacked batches flagged at 1%", flagged, dict(det="veritas_mm", rate=0.01), 0.83, S),
    ("images", "clean batches flagged 3.3% at LOW_RISK. About 4 s", flagged, dict(det="veritas_mm", clean=True), 0.033, C),
    ("images-bb", "80-100% of\nbatches flagged at 5%", flagged, dict(det="veritas_mm", rate=0.05, per_attack="min"), 0.80, S),
    ("images-bb", "80-100% of\nbatches flagged at 5%", flagged, dict(det="veritas_mm", rate=0.05, per_attack="max"), 1.0, S),
    ("images-bb", "40-80% at 1%", flagged, dict(det="veritas_mm", rate=0.01, per_attack="min"), 0.40, S),
    ("images-bb", "40-80% at 1%", flagged, dict(det="veritas_mm", rate=0.01, per_attack="max"), 0.80, S),
    ("images-breadth", "10% of clean\n  batches at LOW_RISK", flagged, dict(det="veritas_mm", clean=True, data="tiny_imagenet"), 0.10, C),
    ("images-breadth", "5.8% at SUSPICIOUS", flagged, dict(det="veritas_mm", clean=True, data="tiny_imagenet", at=SUSPICIOUS), 0.058, C),
    ("label-errors", "(AUROC 0.72-0.92)", label_errors, dict(data="cifar100", stat="auroc", scorer="label", mode="B_no_reference"), 0.72, A),
    ("label-errors", "(AUROC 0.72-0.92)", label_errors, dict(data="mnist", stat="auroc", scorer="label", mode="B_no_reference"), 0.92, A),
    ("label-errors", "only 2-16% of the listed images", label_errors, dict(data="mnist", stat="precision_app_list"), 0.02, S),
    ("label-errors", "only 2-16% of the listed images", label_errors, dict(data="cifar100", stat="precision_app_list", scorer="label", mode="B_no_reference"), 0.16, S),
    ("label-errors", "precision 4% on CIFAR-10", label_errors, dict(data="cifar10", stat="precision_app_list"), 0.04, S),
    ("label-errors", "15% on CIFAR-100", label_errors, dict(data="cifar100", stat="precision_app_list"), 0.15, S),
    ("adaptive", "(14 of 14 with attack success >= 90%) was\n  flagged", adaptive, dict(stat="flagged", min_asr=0.9), 1.0, 0),
    ("adaptive", "at 5% poison it was never\n  flagged", adaptive, dict(stat="flagged", family="cl_lowfreq", rate=0.05), 0.0, 0),
    ("adaptive", "reached 20% attack success", adaptive, dict(stat="asr", family="cl_lowfreq", rate=0.05, s=32), 0.20, S),
    ("audio", "mean row AUPRC 0.74 at 1% and\n0.93 at 5% on Speech Commands", auprc, dict(det="veritas_mm", rate=0.01, data="speech_commands"), 0.74, A),
    ("audio", "mean row AUPRC 0.74 at 1% and\n0.93 at 5% on Speech Commands", auprc, dict(det="veritas_mm", rate=0.05, data="speech_commands"), 0.93, A),
    ("audio", "0.28 and 0.51 on ESC-50", auprc, dict(det="veritas_mm", rate=0.01, data="esc50"), 0.28, A),
    ("audio", "0.28 and 0.51 on ESC-50", auprc, dict(det="veritas_mm", rate=0.05, data="esc50"), 0.51, A),
    ("audio", "clean-label triggers flagged in 19 of 20 Speech Commands", flagged, dict(det="veritas_mm", rate=None, attack="clean_label_backdoor", data="speech_commands"), 0.95, S),
    ("audio", "and\n1 of 20 on ESC-50", flagged, dict(det="veritas_mm", rate=None, attack="clean_label_backdoor", data="esc50"), 0.05, S),
    ("audio", "Speech Commands clean batches flagged 5.0%", flagged, dict(det="veritas_mm", clean=True, data="speech_commands"), 0.05, C),
    ("audio", "ESC-50 clean batches 3.3%", flagged, dict(det="veritas_mm", clean=True, data="esc50"), 0.033, C),
    ("audio-breadth", "clean-label triggers in 6 of 20 batches", flagged, dict(det="veritas_mm", rate=None, attack="clean_label_backdoor"), 0.30, S),
    ("audio-breadth", "clean batches 5.0% at LOW_RISK", flagged, dict(det="veritas_mm", clean=True), 0.05, C),
    ("video", "mean row AUPRC 0.53 at 1% and 0.75 at 5%", auprc, dict(det="veritas_mm", rate=0.01), 0.53, A),
    ("video", "mean row AUPRC 0.53 at 1% and 0.75 at 5%", auprc, dict(det="veritas_mm", rate=0.05), 0.75, A),
    ("video", "46% / 86% of\nattacked batches flagged", flagged, dict(det="veritas_mm", rate=0.01), 0.46, S),
    ("video", "46% / 86% of\nattacked batches flagged", flagged, dict(det="veritas_mm", rate=0.05), 0.86, S),
    ("video", "clean batches flagged 4.2% at LOW_RISK", flagged, dict(det="veritas_mm", clean=True), 0.042, C),
    ("video", "2.5% at SUSPICIOUS (nominal 1%)", flagged, dict(det="veritas_mm", clean=True, at=SUSPICIOUS), 0.025, C),
]


def result_path(run: str, fresh: bool) -> Path:
    stem = RUNS[run][0]
    if not fresh:
        return ROOT / "results" / f"{stem}.json"
    kind, label = _kind_label(stem)
    found = sorted((ROOT / "results").glob(f"*-{kind}-repro-{label}.json"))
    if not found:
        raise FileNotFoundError(f"no rerun for {run}: run `make reproduce ONLY={run}` first")
    return found[-1]


def _kind_label(stem: str) -> tuple[str, str]:
    """2026-11-16-benchmark-audio-test-v3-sig -> ('benchmark-audio', 'test-v3-sig'), from the file's own label."""
    label = json.loads((ROOT / "results" / f"{stem}.json").read_text())["label"]
    return stem[11:-len(label) - 1], label


def command(run: str) -> str:
    stem, env, mod = RUNS[run]
    return f"{env} python -m research.{mod} --label repro-{_kind_label(stem)[1]}".strip()


def rerun(runs: list[str]) -> None:
    """Run each benchmark from its manifest into a repro-* result file; stops at the first failure."""
    for run in runs:
        stem, env, mod = RUNS[run]
        print("+", command(run), flush=True)
        subprocess.run([sys.executable, "-m", f"research.{mod.split()[0]}", *mod.split()[1:], "--label", f"repro-{_kind_label(stem)[1]}"],
                       env={**os.environ, "PYTHONPATH": str(ROOT.parent), **dict(kv.split("=", 1) for kv in env.split())},
                       cwd=ROOT.parent, check=True)


def check(fresh: bool, only: list[str] | None) -> int:
    card = " ".join(CARD.read_text().split())
    cache, bad, n = {}, 0, 0
    for run, text, fn, kw, want, tol in CLAIMS:
        if only and run not in only:
            continue
        n += 1
        flat = " ".join(text.split())
        if flat not in card:
            print(f"CARD  {run:15} text not in model card: {flat!r}")
            bad += 1
            continue
        try:
            if run not in cache:
                cache[run] = json.loads(result_path(run, fresh).read_text())
            got = fn(cache[run], **kw)
        except (FileNotFoundError, KeyError, ValueError, statistics.StatisticsError) as exc:
            print(f"ERROR {run:15} {flat[:60]!r}: {exc}")
            bad += 1
            continue
        ok = abs(got - want) <= tol + 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {run:15} {got:.3f} (claimed {want:g} +-{tol:g})  {flat[:70]}")
    if fresh:
        for run in cache:
            print(f"      {run:15} detector {cache[run].get('detector_commit', '?')}  manifest {cache[run]['manifest']['benchmark_id']}")
    print(f"{n - bad} ok, {bad} failed")
    return 1 if bad else 0


def table() -> str:
    """One row per run: the claimed numbers it backs, and its result file."""
    rows = ["| Run | Claims it backs (numbers in the text above) | Result file |", "|---|---|---|"]
    for run, (stem, *_) in RUNS.items():
        shown = []
        for r, text, *_ in CLAIMS:
            cells = [c.strip(" *") for c in " ".join(text.split()).strip("|").split("|")]
            t = f"{cells[0]} {cells[-1]}" if len(cells) > 1 else cells[0]
            if r == run and t not in shown:
                shown.append(t)
        rows.append(f"| `{run}` | {'; '.join(shown)} | [`{stem}`](../backend/research/results/{stem}.md) |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fresh", action="store_true", help="check the repro-* reruns instead of the committed files")
    p.add_argument("--only", nargs="+", choices=list(RUNS))
    p.add_argument("--rerun", action="store_true", help="rerun the benchmarks first, then check the reruns")
    p.add_argument("--commands", action="store_true")
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    if a.commands:
        print("\n".join(f"{r}\t{command(r)}" for r in (a.only or RUNS)))
        return 0
    if a.table:
        print(table())
        return 0
    if a.rerun:
        rerun(a.only or list(RUNS))
    return check(a.fresh or a.rerun, a.only)


if __name__ == "__main__":
    sys.exit(main())
