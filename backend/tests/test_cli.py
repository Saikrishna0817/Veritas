"""The command-line client against a real server on a real socket, authenticated with an API key."""
import os
import pickle
import socket
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
import veritas_cli  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def server():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True); t.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/api/v1"
    srv.should_exit = True; t.join(timeout=5)


@pytest.fixture(scope="module")
def key(server):
    from app.core.security import create_access_token
    import urllib.request, json
    from app.models import database as db
    uid = db.create_user("cli-user", "x", role="user", email="cli@example.com")      # keys belong to real accounts
    token = create_access_token({"id": uid, "name": "cli-user", "role": "user"})
    req = urllib.request.Request(f"{server}/account/api-keys", data=json.dumps({"name": "ci"}).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["key"]


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo x",))


def test_scan_model_exit_codes(server, key, tmp_path, capsys):
    bad, ok = tmp_path / "bad.pkl", tmp_path / "ok.pkl"
    bad.write_bytes(pickle.dumps(_Evil())); ok.write_bytes(pickle.dumps({"w": [1, 2]}))
    assert veritas_cli.main(["--api-url", server, "--api-key", key, "scan-model", str(bad)]) == 1
    assert "DANGEROUS" in capsys.readouterr().out
    assert veritas_cli.main(["--api-url", server, "--api-key", key, "scan-model", str(ok)]) == 0


def test_check_data_and_error_paths(server, key, tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(500, 5); y = (X[:, 0] + X[:, 1] > 0).astype(int); y[350:410] = 1 - y[350:410]
    f = tmp_path / "d.csv"; pd.DataFrame(X, columns=list("abcde")).assign(label=y).to_csv(f, index=False)
    assert veritas_cli.main(["--api-url", server, "--api-key", key, "check-data", str(f)]) == 1
    assert veritas_cli.main(["--api-url", server, "--api-key", "vk_wrong", "scan-model", str(f)]) == 2
    assert veritas_cli.main(["--api-url", "http://127.0.0.1:9", "--api-key", key, "scan-model", str(f)]) == 2
