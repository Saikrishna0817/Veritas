"""Blob storage for uploads that wait for a worker. ``file:///dir`` (default) or ``s3://bucket`` (S3, Cloudflare R2,
MinIO via VERITAS_S3_ENDPOINT). Keys are generated server-side; callers never pass user-controlled paths."""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.core.config import settings


def new_key(owner_id: str, kind: str = "upload") -> str:
    return f"{kind}/{uuid.uuid4().hex}"        # not derived from the owner or file name: nothing to traverse or guess


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("bad storage key")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage:
    def __init__(self, bucket: str, endpoint: str = "") -> None:
        import boto3
        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint or None)

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def make_storage(url: str = "", endpoint: str = ""):
    url = url or settings.storage_url or f"file://{os.getenv('VERITAS_UPLOAD_DIR', os.path.join(tempfile.gettempdir(), 'veritas-uploads'))}"
    if url.startswith("s3://"):
        return S3Storage(url.removeprefix("s3://").strip("/"), endpoint or settings.s3_endpoint)
    return LocalStorage(url.removeprefix("file://"))


_storage = None


def storage():
    global _storage
    if _storage is None:
        _storage = make_storage()
    return _storage
