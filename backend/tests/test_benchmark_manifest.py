import json
from pathlib import Path


MANIFEST = Path(__file__).parents[1] / "research" / "benchmark" / "manifest.json"


def test_benchmark_manifest_is_versioned_and_complete():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["immutable"] is True
    assert manifest["random_seed"] == 20260902
    assert {attack["name"] for attack in manifest["attacks"]} == {
        "label_flip", "backdoor", "clean_label", "gradient_poisoning", "boiling_frog",
    }
    assert all(attack["poison_count"] > 0 for attack in manifest["attacks"])
