"""Runs picklescan and modelscan on every corpus file. Execute with the Python that has them installed:

    python comparator_worker.py CORPUS_ROOT OUT.json

Stdlib + those two packages only. Neither tool loads the files it scans.
"""
import json
import sys
import time
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads((root / "manifest.json").read_text())["items"]
result = {"picklescan": {}, "modelscan": {}}

from picklescan.scanner import scan_file_path
from modelscan.modelscan import ModelScan
from modelscan.settings import DEFAULT_SETTINGS

for i, item in enumerate(items):
    path = root / item["path"]
    t = time.perf_counter()
    try:
        r = scan_file_path(str(path))
        result["picklescan"][item["path"]] = {"flagged": r.infected_files > 0 or bool(r.issues_count), "error": bool(r.scanned_files == 0 and not r.infected_files),
                                              "seconds": time.perf_counter() - t, "detail": [g.__dict__.get("name") or str(g) for g in r.globals][:5]}
    except Exception as exc:
        result["picklescan"][item["path"]] = {"flagged": False, "error": True, "seconds": time.perf_counter() - t, "detail": str(exc)[:200]}
    t = time.perf_counter()
    try:
        ms = ModelScan(settings=DEFAULT_SETTINGS)
        ms.scan(path)
        issues = ms.issues.all_issues
        result["modelscan"][item["path"]] = {"flagged": bool(issues), "skipped": bool(ms.skipped), "error": bool(ms.errors),
                                             "seconds": time.perf_counter() - t, "detail": [str(x.details) for x in issues][:3]}
    except Exception as exc:
        result["modelscan"][item["path"]] = {"flagged": False, "skipped": False, "error": True, "seconds": time.perf_counter() - t, "detail": str(exc)[:200]}
    if i % 50 == 0:
        print(i, len(items), flush=True)
out.write_text(json.dumps(result))
