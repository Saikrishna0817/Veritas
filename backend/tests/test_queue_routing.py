"""Untrusted binary formats (images, audio, video, model files) must only ever be queued for the isolated media worker;
CSV jobs stay on the ordinary queue. Checked without a Redis server: building an RQ queue does not connect."""
import dataclasses

import pytest

from app.jobs import queue as jobs_queue


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(jobs_queue, "settings", dataclasses.replace(jobs_queue.settings, redis_url="redis://127.0.0.1:1/0"))
    return jobs_queue.JobQueue()


@pytest.mark.parametrize("job_type", ["image_upload", "audio_upload", "video_upload", "model_scan"])
def test_media_and_model_jobs_go_to_the_isolated_queue(store, job_type):
    assert job_type in jobs_queue.UNTRUSTED_JOBS
    assert store._redis_queue(job_type).name == "veritas-untrusted"


def test_table_jobs_stay_on_the_ordinary_queue(store):
    assert "csv_upload" not in jobs_queue.UNTRUSTED_JOBS
    assert store._redis_queue("csv_upload").name == "veritas"


def test_the_upload_routes_submit_the_untrusted_job_types():
    """The routes name the job types the routing above relies on (a rename would silently move them)."""
    import inspect

    from app.api.routes import models, upload
    src = inspect.getsource(upload) + inspect.getsource(models)
    for job_type in ("image_upload", "audio_upload", "video_upload", "model_scan"):
        assert f'"{job_type}"' in src
