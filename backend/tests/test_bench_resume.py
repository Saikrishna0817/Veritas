"""A benchmark run killed part-way resumes from its checkpoint without redoing finished batches."""
import pytest

bench_mm = pytest.importorskip("research.bench_mm")

SPECS = [{"split": "test", "data_id": "d", "seed": 0, "attack": a, "rate": r, "rep": 0}
         for a in (None, "x", "y") for r in (0.01, 0.05)]


def _score(job):
    return dict(job["spec"])


def test_killed_run_resumes_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mm, "score_batch", _score)
    ckpt, prepared = tmp_path / "ckpt.pkl", []

    def jobs(m, done, die_after=None):
        for s in SPECS:
            if bench_mm.spec_key(s) in done:
                continue
            if die_after is not None and len(prepared) == die_after:
                raise MemoryError("killed")
            prepared.append(bench_mm.spec_key(s))
            yield {"spec": s}

    with pytest.raises(MemoryError):
        bench_mm.score_all({}, lambda m, done: jobs(m, done, die_after=3), len(SPECS), 1, ckpt)
    saved = sum(1 for _ in _records(ckpt))
    prepared.clear()
    records = bench_mm.score_all({}, jobs, len(SPECS), 1, ckpt)

    assert sorted(map(bench_mm.spec_key, records), key=str) == sorted(map(bench_mm.spec_key, SPECS), key=str)
    assert 1 <= saved <= 3 and len(prepared) == len(SPECS) - saved   # saved batches are not prepared again


def _records(path):
    import pickle
    with path.open("rb") as f:
        while True:
            try: yield pickle.load(f)
            except EOFError: return
