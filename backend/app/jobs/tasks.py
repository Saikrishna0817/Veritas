"""Job bodies. They run in the worker (or the in-process pool) and read their input from blob storage, so the web
process never holds an upload while it waits for CPU."""
from __future__ import annotations

from app.core.storage import storage
from app.jobs.queue import TASKS


def csv_upload(payload: dict, owner_id: str) -> dict:
    from app.api import dependencies as deps
    from app.api.analysis_helpers import _orchestrate_csv

    store = storage()
    split: dict = {}
    keys = [payload["key"]] + ([payload["reference_key"]] if payload.get("reference_key") else [])
    try:
        content = store.get(payload["key"])
        if payload.get("reference_key"):
            split["reference_bytes"] = store.get(payload["reference_key"])
        if payload.get("split_column"):
            split.update(split_column=payload["split_column"], split_cutoff=payload["split_cutoff"])
        result, dataset_id = _orchestrate_csv(content, payload["filename"], owner_id, split)
    finally:
        for key in keys:                       # uploads are kept only as long as the job needs them
            store.delete(key)
    result["source"], result["owner_id"] = "upload", owner_id
    result = deps.to_serializable(result)
    deps.db.save_result(result, "upload", payload["filename"], owner_id=owner_id)
    deps.db.log_audit_event(owner_id, "analysis.uploaded", "dataset", dataset_id,
                            {"filename": payload["filename"], "n_samples": result.get("n_samples"), "verdict": result.get("verdict")})
    return result


def model_scan(payload: dict, owner_id: str) -> dict:
    from app.api import dependencies as deps
    from app.api.analysis_helpers import _orchestrate_model

    store = storage()
    try:
        data = store.get(payload["key"])
    finally:
        store.delete(payload["key"])
    result = deps.to_serializable(_orchestrate_model(data, payload["filename"]))
    result["owner_id"] = owner_id
    deps.db.save_model_scan(result, owner_id)
    deps.db.log_audit_event(owner_id, "analysis.model_scanned", "model", result["scan_id"],
                            {"filename": payload["filename"], "format": result["scan"]["format"], "verdict": result["verdict"]})
    return result


def image_upload(payload: dict, owner_id: str) -> dict:
    from app.api import dependencies as deps
    from app.api.analysis_helpers import _orchestrate_images

    store = storage()
    keys = [payload["key"]] + ([payload["reference_key"]] if payload.get("reference_key") else [])
    try:
        reference = store.get(payload["reference_key"]) if payload.get("reference_key") else None
        result = _orchestrate_images(store.get(payload["key"]), payload["filename"], owner_id, reference)
    finally:
        for key in keys:
            store.delete(key)
    result["owner_id"] = owner_id
    result = deps.to_serializable(result)
    deps.db.save_result(result, "image_upload", payload["filename"], owner_id=owner_id)
    deps.db.log_audit_event(owner_id, "analysis.images_uploaded", "dataset", result["dataset_info"]["dataset_id"],
                            {"filename": payload["filename"], "n_images": result["dataset_info"]["n_images"], "verdict": result["verdict"]})
    return result


TASKS["csv_upload"] = csv_upload
TASKS["image_upload"] = image_upload
TASKS["model_scan"] = model_scan
