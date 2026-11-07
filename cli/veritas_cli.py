#!/usr/bin/env python3
"""veritas: scan a model file or audit a dataset from a terminal or CI job. Standard library only.

  export VERITAS_API_URL=https://veritas.example.com/api/v1      # default http://localhost:8001/api/v1
  export VERITAS_API_KEY=vk_...                                  # create one under Account > API keys
  veritas scan-model model.pkl                    # exit 1 if the file can run code (DANGEROUS)
  veritas scan-model model.pkl --fail-on review   # also fail on REVIEW
  veritas check-data data.csv --reference trusted.csv --fail-on SUSPICIOUS

Exit codes: 0 passed, 1 verdict at or above --fail-on, 2 usage / network / server error.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

MODEL_ORDER = ["SAFE_FORMAT", "NO_CODE_EXECUTION_FOUND", "REVIEW", "DANGEROUS"]
DATA_ORDER = ["CLEAN", "LOW_RISK", "SUSPICIOUS", "CONFIRMED_POISONED"]


class CliError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _multipart(files: dict[str, str], fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, path in files.items():
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{os.path.basename(path)}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
        with open(path, "rb") as fh:
            out.append(fh.read() + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def _call(url: str, key: str, body: bytes | None = None, ctype: str | None = None, method: str | None = None) -> dict:
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body is not None else "GET"))
    req.add_header("Authorization", f"Bearer {key}")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except Exception:
            detail = exc.reason
        raise CliError(f"server said {exc.code}: {detail}", exc.code) from None
    except urllib.error.URLError as exc:
        raise CliError(f"cannot reach {url}: {exc.reason}") from None


def _post_json(url: str, key: str, payload: dict) -> dict:
    return _call(url, key, json.dumps(payload).encode(), "application/json")


def _direct(base: str, key: str, path: str, files: dict[str, str], fields: dict[str, str]) -> dict:
    """Upload each file straight to object storage with a presigned URL, then ask the API to queue the job."""
    roles = {"file": "csv", "reference_file": "csv_reference", "model_file": "model"}
    ids = {}
    for name, fpath in files.items():
        with open(fpath, "rb") as fh:
            data = fh.read()
        grant = _post_json(f"{base}/uploads/presign", key, {"role": roles[name], "filename": os.path.basename(fpath), "size": len(data)})
        put = urllib.request.Request(grant["url"], data=data, method="PUT")      # no Authorization header: the URL is the credential
        try:
            urllib.request.urlopen(put, timeout=600).read()
        except urllib.error.HTTPError as exc:
            raise CliError(f"storage refused the upload: {exc.code} {exc.reason}") from None
        except urllib.error.URLError as exc:
            raise CliError(f"cannot reach storage: {exc.reason}") from None
        ids[name] = grant["upload_id"]
    done = {"upload_id": ids.get("file") or ids["model_file"]}
    if "reference_file" in ids:
        done["reference_upload_id"] = ids["reference_file"]
    done.update({k: v for k, v in fields.items() if k in ("split_column", "split_cutoff")})
    return _post_json(f"{base}/uploads/complete", key, done)


def _run_job(base: str, key: str, path: str, files: dict[str, str], fields: dict[str, str], wait: float) -> dict:
    try:
        job = _direct(base, key, path, files, fields)
    except CliError as exc:
        if exc.status != 501:                    # 501: the server stores uploads locally, so send the bytes through the API instead
            raise
        body, ctype = _multipart(files, fields)
        job = _call(f"{base}{path}?background=true", key, body, ctype)
    deadline = time.time() + wait
    while time.time() < deadline:
        state = _call(f"{base}/jobs/{job['job_id']}", key)
        if state["status"] == "completed":
            return state["result"]
        if state["status"] == "failed":
            raise CliError(f"analysis failed: {state['error']}")
        time.sleep(1.5)
    raise CliError(f"timed out after {wait:.0f}s waiting for job {job['job_id']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="veritas", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=os.getenv("VERITAS_API_URL", "http://localhost:8001/api/v1"))
    ap.add_argument("--api-key", default=os.getenv("VERITAS_API_KEY"))
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--wait", type=float, default=600, help="seconds to wait for the analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("scan-model", help="statically scan a model file for code execution")
    m.add_argument("file")
    m.add_argument("--fail-on", default="dangerous", choices=["dangerous", "review"])
    d = sub.add_parser("check-data", help="audit a CSV for wrong labels and shifted rows")
    d.add_argument("file")
    d.add_argument("--reference", help="trusted CSV with the same columns; the main file is then all incoming")
    d.add_argument("--split-column")
    d.add_argument("--split-cutoff")
    d.add_argument("--fail-on", default="SUSPICIOUS", choices=DATA_ORDER[1:])
    args = ap.parse_args(argv)
    if not args.api_key:
        print("error: set VERITAS_API_KEY (or --api-key)", file=sys.stderr)
        return 2
    base = args.api_url.rstrip("/")
    try:
        if args.cmd == "scan-model":
            result = _run_job(base, args.api_key, "/analyze/model", {"model_file": args.file}, {}, args.wait)
            verdict, order, limit = result["verdict"], MODEL_ORDER, "DANGEROUS" if args.fail_on == "dangerous" else "REVIEW"
            for f in result["scan"]["findings"][:20]:
                print(f"  [{f['severity']}] {f['detail']}")
        else:
            files, fields = {"file": args.file}, {}
            if args.reference:
                files["reference_file"] = args.reference
            if args.split_column:
                fields.update(split_column=args.split_column, split_cutoff=args.split_cutoff or "")
            result = _run_job(base, args.api_key, "/analyze/upload", files, fields, args.wait)
            verdict, order, limit = result["verdict"], DATA_ORDER, args.fail_on
            flagged = result["layer_results"]["row_flags"]["flagged_indices_incoming"]
            print(f"  {len(flagged)} of {result['n_samples']} incoming rows flagged; suspicion {result['overall_suspicion_score']:.2f}")
    except (CliError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    print(f"{args.cmd}: {args.file}: {verdict}")
    return 1 if order.index(verdict) >= order.index(limit) else 0


if __name__ == "__main__":
    sys.exit(main())
