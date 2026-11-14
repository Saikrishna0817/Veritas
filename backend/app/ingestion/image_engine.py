"""Image (and, through ingest_zip, audio) dataset ingestion: a zip with one folder per class (``cat/001.png``), optionally a second zip of trusted
reference images in the same layout. Images are decoded in memory (nothing is extracted to disk), converted to RGB
and resized to 32x32, the size the image detector was benchmarked at (docs/image-benchmark-findings.md)."""
from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath

import numpy as np

SIZE = 32
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
MAX_IMAGES = 6000            # reference + incoming; the shared-pattern checks build an n x n similarity matrix
MAX_ENTRIES = 50_000         # zip members looked at
MAX_UNCOMPRESSED = 1 << 30   # 1 GB of image bytes in total: a zip bomb stops here
MAX_FILE = 20 << 20          # 20 MB per image
MAX_PIXELS = 50_000_000      # per decoded image (Pillow's own bomb guard is only a warning by default)
MIN_REFERENCE, MIN_INCOMING = 100, 20
REFERENCE_FRACTION = 0.7


def _decode_image(raw: bytes) -> np.ndarray:
    """Image bytes -> 32x32x3 uint8 (raises on corrupt, unsupported, or over MAX_PIXELS)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)


def _read_zip(data: bytes, what: str, warnings: list[str], extensions=EXTENSIONS, decode=_decode_image,
              noun: str = "image") -> list[tuple[str, str, np.ndarray]]:
    """[(path, class name, decoded)] in path order. Unreadable or unsafe members are skipped and counted."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"The {what} is not a valid zip file.") from exc
    members = [m for m in zf.infolist() if not m.is_dir()]
    if len(members) > MAX_ENTRIES:
        raise ValueError(f"The {what} has {len(members)} files; the limit is {MAX_ENTRIES}.")
    images, skipped, total = [], 0, 0
    for m in sorted(members, key=lambda m: m.filename):
        path = PurePosixPath(m.filename.replace("\\", "/"))
        if any(p.startswith(".") or p == "__MACOSX" for p in path.parts) or path.suffix.lower() not in extensions:
            continue                                   # system files and non-images are not counted as skipped
        if m.file_size > MAX_FILE:
            skipped += 1
            continue
        total += m.file_size
        if total > MAX_UNCOMPRESSED:
            raise ValueError(f"The {what} unpacks to more than {MAX_UNCOMPRESSED >> 20} MB of {noun}s.")
        try:
            arr = decode(zf.read(m))
        except Exception:      # corrupt, truncated, unsupported, too large
            skipped += 1
            continue
        images.append((str(path), path.parent.name if len(path.parts) > 1 else "", arr))
    if skipped:
        warnings.append(f"Skipped {skipped} {noun}(s) in the {what} that could not be read or were over {MAX_FILE >> 20} MB.")
    return images


def ingest_zip(zip_bytes: bytes, filename: str, reference_bytes: bytes | None, extensions, decode, noun: str, stack) -> dict:
    """Class-folder zip (plus an optional reference zip) -> stacked items, integer labels, reference split. Shared by
    the image and audio engines; ``stack`` turns the decoded list into one array."""
    warnings: list[str] = []
    main = _read_zip(zip_bytes, "zip file", warnings, extensions, decode, noun)
    ref = _read_zip(reference_bytes, "reference zip", warnings, extensions, decode, noun) if reference_bytes is not None else None
    rng = np.random.default_rng(0)
    if ref is None:
        # Images have no natural order (zips list them class by class), so the reference is a random, per-class
        # 70% rather than the first 70% in the file.
        order = rng.permutation(len(main))
        classes_of = np.array([c for _, c, _ in main])
        is_ref = np.zeros(len(main), bool)
        for c in np.unique(classes_of):
            idx = order[classes_of[order] == c]
            is_ref[idx[:int(round(len(idx) * REFERENCE_FRACTION))]] = True
        ref, main = [main[i] for i in np.flatnonzero(is_ref)], [main[i] for i in np.flatnonzero(~is_ref)]
        warnings.append(f"Reference = a random {REFERENCE_FRACTION:.0%} of each class; incoming = the rest. If bad {noun}s may be "
                        "anywhere in the zip, upload a separate trusted reference zip instead.")
    else:
        ref = [ref[i] for i in rng.permutation(len(ref))]   # the detector takes its calibration slice from the front
    if len(ref) + len(main) > MAX_IMAGES:
        keep_ref = min(len(ref), max(MIN_REFERENCE, MAX_IMAGES - min(len(main), MAX_IMAGES // 3)))
        keep_inc = MAX_IMAGES - keep_ref
        warnings.append(f"Using {keep_ref} reference and {keep_inc} incoming {noun}s of {len(ref)} and {len(main)} (limit {MAX_IMAGES}).")
        ref = [ref[i] for i in sorted(rng.choice(len(ref), keep_ref, replace=False))]
        main = [main[i] for i in sorted(rng.choice(len(main), min(keep_inc, len(main)), replace=False))]
    if len(ref) < MIN_REFERENCE or len(main) < MIN_INCOMING:
        raise ValueError(f"Need at least {MIN_REFERENCE} reference and {MIN_INCOMING} incoming {noun}s; got {len(ref)} and {len(main)}.")
    names = sorted({c for _, c, _ in ref + main})
    has_labels = len(names) > 1
    if not has_labels:
        warnings.append(f"{noun.capitalize()}s are not in class folders (or there is one class): label check skipped.")
    code = {c: i for i, c in enumerate(names)}
    rows = ref + main
    return {
        "dataset_id": str(uuid.uuid4()), "filename": filename,
        "items": stack([a for _, _, a in rows], warnings), "labels": np.array([code[c] for _, c, _ in rows]),
        "paths": [p for p, _, _ in rows], "class_names": names, "has_labels": has_labels,
        "reference_split": len(ref), "n_items": len(rows), "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class ImageIngestionEngine:
    def ingest(self, zip_bytes: bytes, filename: str, reference_bytes: bytes | None = None) -> dict:
        out = ingest_zip(zip_bytes, filename, reference_bytes, EXTENSIONS, _decode_image, "image", lambda a, w: np.stack(a))
        out["images"], out["n_images"] = out.pop("items"), out.pop("n_items")
        return out
