"""Dataset/model helpers."""

from __future__ import annotations

from typing import Any, Dict

from app.demo.data_generator import get_demo_data
from app.demo.real_datasets import DATASET_CATALOG, get_real_dataset


def demo_dataset() -> Dict[str, Any]:
    return get_demo_data()


def list_real_datasets() -> Dict[str, Any]:
    return {"datasets": DATASET_CATALOG}


def load_real_dataset(name: str) -> Dict[str, Any]:
    return get_real_dataset(name)

