"""Score the Veritas model-file scanner against picklescan and modelscan on the scanner-bench corpus.

    python -m research.scanner_bench.run_scanner_bench --comparator-python /path/to/python-with-modelscan-and-picklescan [--label r1]

Run build_corpus first. Results go to research/results/<date>-scanner-bench-<label>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from app.scanner.model_scan import scan_bytes
from research.scanner_bench.build_corpus import ROOT

RESULTS = Path(__file__).parents[1] / "results"
HERE = Path(__file__).parent


def run_veritas(items):
    out = {}
    for it in items:
        data = (ROOT / it["path"]).read_bytes()
        t = time.perf_counter()
        r = scan_bytes(data, Path(it["path"]).name)
        out[it["path"]] = {"verdict": r["verdict"], "seconds": time.perf_counter() - t, "format": r["format"],
                           "detail": [f["kind"] for f in r["findings"]][:4]}
    return out


def rate(hits, n):
    return f"{hits}/{n} ({100 * hits / n:.0f}%)" if n else "-"


def summarise(items, veritas, comp, split=None):
    items = [i for i in items if split is None or i.get("split", "dev") == split]
    tools = {
        "veritas (DANGEROUS)": lambda p: veritas[p]["verdict"] == "DANGEROUS",
        "veritas (DANGEROUS or REVIEW)": lambda p: veritas[p]["verdict"] in ("DANGEROUS", "REVIEW"),
        "picklescan": lambda p: comp["picklescan"][p]["flagged"],
        "modelscan": lambda p: comp["modelscan"][p]["flagged"],
    }
    mal = [i for i in items if i["label"] == "malicious"]
    ben = [i for i in items if i["label"] == "benign"]
    table = {}
    for name, flagged in tools.items():
        table[name] = {"recall": sum(flagged(i["path"]) for i in mal), "n_malicious": len(mal),
                       "false_positives": sum(flagged(i["path"]) for i in ben), "n_benign": len(ben)}
    by_family = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for i in mal:
        for name, flagged in tools.items():
            c = by_family[i["family"]][name]
            c[0] += flagged(i["path"]); c[1] += 1
    by_variant = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for i in mal:
        for name, flagged in tools.items():
            c = by_variant[i["variant"]][name]
            c[0] += flagged(i["path"]); c[1] += 1
    fps = {name: [i["path"] for i in ben if flagged(i["path"])] for name, flagged in tools.items()}
    secs = {"veritas": sum(v["seconds"] for v in veritas.values()), "picklescan": sum(v["seconds"] for v in comp["picklescan"].values()),
            "modelscan": sum(v["seconds"] for v in comp["modelscan"].values())}
    return table, by_family, by_variant, fps, secs


def to_markdown(label, table, by_family, by_variant, fps, secs, items):
    names = list(table)
    md = [f"# Scanner benchmark: {label}", "",
          f"{table[names[0]]['n_malicious']} generated malicious files, {table[names[0]]['n_benign']} benign files "
          "(public Hugging Face models at pinned revisions plus locally trained models). Files are scanned, never loaded.", "",
          "| scanner | malicious caught | benign flagged |", "|---|---|---|"]
    for n in names:
        t = table[n]
        md.append(f"| {n} | {rate(t['recall'], t['n_malicious'])} | {rate(t['false_positives'], t['n_benign'])} |")
    md += ["", f"Total scan time over the corpus (s): " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()), "", "## By payload family (caught/total)", "",
           "| family | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for fam in sorted(by_family):
        md.append(f"| {fam} | " + " | ".join(f"{by_family[fam][n][0]}/{by_family[fam][n][1]}" for n in names) + " |")
    md += ["", "## By container variant (caught/total)", "", "| variant | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for v in sorted(by_variant):
        md.append(f"| {v} | " + " | ".join(f"{by_variant[v][n][0]}/{by_variant[v][n][1]}" for n in names) + " |")
    md += ["", "## Benign files flagged", ""]
    for n in names:
        md.append(f"- **{n}**: " + (", ".join(Path(p).name for p in fps[n][:25]) or "none") + (f" (+{len(fps[n]) - 25} more)" if len(fps[n]) > 25 else ""))
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator-python", required=True)
    ap.add_argument("--label", default="r1")
    args = ap.parse_args()
    items = json.loads((ROOT / "manifest.json").read_text())["items"]
    veritas = run_veritas(items)
    comp_path = ROOT / "comparators.json"
    subprocess.run([args.comparator_python, str(HERE / "comparator_worker.py"), str(ROOT), str(comp_path)], check=True)
    comp = json.loads(comp_path.read_text())
    out = RESULTS / f"{date.today().isoformat()}-scanner-bench-{args.label}"
    md, dump = [], {"veritas": veritas, "comparators": comp}
    for split in ("dev", "heldout"):
        table, by_family, by_variant, fps, secs = summarise(items, veritas, comp, split)
        md.append(to_markdown(f"{args.label} / {split}", table, by_family, by_variant, fps, secs, items))
        dump[split] = {"table": table, "by_family": by_family, "by_variant": by_variant, "false_positives": fps, "seconds": secs}
    out.with_suffix(".json").write_text(json.dumps(dump, indent=1, default=dict))
    out.with_suffix(".md").write_text("\n".join(md))
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
