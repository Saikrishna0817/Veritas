"""Reproducible research utilities for Veritas."""


def detector_commit() -> str:
    """The commit a benchmark ran at ('-dirty' if the tree had uncommitted changes), recorded in every result."""
    import subprocess
    from pathlib import Path
    return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                          cwd=Path(__file__).parent).stdout.strip() or "unknown"
