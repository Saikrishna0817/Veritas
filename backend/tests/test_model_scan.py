"""The static model-file scanner: it must flag code-running files without ever executing them."""
import gzip
import io
import json
import os
import pickle
import struct
import zipfile

import numpy as np
import pytest

from app.scanner.model_scan import scan_bytes, scan_sandboxed

FLAG = "VERITAS_SCAN_EXECUTED"


class _Eval:
    def __reduce__(self):
        return (eval, (f"__import__('os').environ.__setitem__('{FLAG}', '1')",))


class _System:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


class _Getattr:   # builtins.getattr gadget chain
    def __reduce__(self):
        return (getattr, ("abc", "__class__"))


class Unknown:    # a class from a module the scanner has never heard of
    pass


def _kinds(result):
    return {f["kind"] for f in result["findings"]}


def test_scanning_never_executes_the_payload(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    r = scan_bytes(pickle.dumps(_Eval()), "evil.pkl")
    assert r["verdict"] == "DANGEROUS" and FLAG not in os.environ


@pytest.mark.parametrize("payload", [_Eval(), _System(), _Getattr()])
@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
def test_dangerous_pickles_are_flagged_at_every_protocol(payload, protocol):
    r = scan_bytes(pickle.dumps(payload, protocol=protocol), "evil.pkl")
    assert r["verdict"] == "DANGEROUS" and r["can_run_code_on_load"] is True
    assert "dangerous_import" in _kinds(r)


def test_payload_appended_after_a_clean_pickle_is_still_found():
    clean = pickle.dumps({"weights": [1, 2, 3]})
    r = scan_bytes(clean + pickle.dumps(_System()), "model.pkl")
    assert r["verdict"] == "DANGEROUS"


def test_compression_does_not_hide_a_payload():
    r = scan_bytes(gzip.compress(pickle.dumps(_System())), "model.joblib")
    assert r["verdict"] == "DANGEROUS" and r["compression"] == "gzip"


def test_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr("app.scanner.model_scan.MAX_DECOMPRESSED", 1024 * 1024)
    r = scan_bytes(gzip.compress(b"\0" * (4 * 1024 * 1024)), "bomb.pkl")
    assert r["verdict"] == "REVIEW" and "scan_error" in _kinds(r)


def test_unknown_import_needs_review_not_a_pass():
    r = scan_bytes(pickle.dumps(Unknown()), "m.pkl")
    assert r["verdict"] == "REVIEW" and "unknown_import" in _kinds(r)


def test_sklearn_model_has_no_code_execution_found():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.RandomState(0)
    X = rng.rand(80, 4)
    model = RandomForestClassifier(n_estimators=3).fit(X, X[:, 0] > 0.5)
    r = scan_bytes(pickle.dumps(model), "rf.pkl")
    assert r["verdict"] == "NO_CODE_EXECUTION_FOUND", r["findings"]
    assert r["can_run_code_on_load"] is False


def test_garbage_is_not_waved_through():
    r = scan_bytes(b"\x80\x04\xff\xff not a pickle", "m.pkl")
    assert r["verdict"] == "REVIEW"
    assert scan_bytes(b"hello world", "m.bin")["verdict"] == "REVIEW"


def _safetensors(tensors=None, header_len=None):
    header = json.dumps(tensors if tensors is not None else {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return struct.pack("<Q", header_len or len(header)) + header + b"\0" * 8


def test_valid_safetensors_is_a_safe_format():
    r = scan_bytes(_safetensors(), "model.safetensors")
    assert r["verdict"] == "SAFE_FORMAT" and r["can_run_code_on_load"] is False


def test_malformed_safetensors_is_flagged():
    assert scan_bytes(_safetensors(header_len=10 ** 12), "m.safetensors")["verdict"] != "SAFE_FORMAT"
    bad = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 800]}}
    assert scan_bytes(_safetensors(bad), "m.safetensors")["verdict"] == "REVIEW"


def test_torch_zip_with_clean_state_dict_and_with_payload():
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.save({"w": torch.zeros(3)}, buf)
    clean = scan_bytes(buf.getvalue(), "m.pt")
    assert clean["format"] == "pytorch_zip" and clean["verdict"] == "NO_CODE_EXECUTION_FOUND", clean["findings"]

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(_System()))
    assert scan_bytes(evil.getvalue(), "m.pt")["verdict"] == "DANGEROUS"


def test_legacy_torch_format_is_scanned():
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.save({"w": torch.zeros(3)}, buf, _use_new_zipfile_serialization=False)
    r = scan_bytes(buf.getvalue(), "old.pt")
    assert r["format"] == "pickle" and r["verdict"] == "NO_CODE_EXECUTION_FOUND", r["findings"]


def test_zip_with_executable_member_is_dangerous():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("model/weights.npy", b"x")
        z.writestr("model/run.sh", "rm -rf /")
    assert scan_bytes(b.getvalue(), "m.zip")["verdict"] == "DANGEROUS"


def test_npy_numeric_is_safe_and_object_array_is_not():
    ok = io.BytesIO(); np.save(ok, np.arange(5))
    assert scan_bytes(ok.getvalue(), "a.npy")["verdict"] == "SAFE_FORMAT"
    evil = io.BytesIO()
    arr = np.empty(1, dtype=object); arr[0] = _System()
    np.save(evil, arr, allow_pickle=True)
    assert scan_bytes(evil.getvalue(), "a.npy")["verdict"] == "DANGEROUS"


def _pb(field, wire, payload):
    def varint(n):
        out = b""
        while True:
            b = n & 0x7F; n >>= 7
            out += bytes([b | (0x80 if n else 0)])
            if not n:
                return out
    key = varint(field << 3 | wire)
    return key + (varint(len(payload)) + payload if wire == 2 else varint(payload))


def _onnx(domain="", external=None):
    node = _pb(4, 2, b"Relu") + (_pb(7, 2, domain.encode()) if domain else b"")
    graph = _pb(1, 2, node)
    if external:
        entry = _pb(1, 2, b"location") + _pb(2, 2, external.encode())
        graph += _pb(5, 2, _pb(13, 2, entry))
    return _pb(1, 0, 8) + _pb(7, 2, graph)


def test_onnx_standard_custom_operator_and_external_path():
    assert scan_bytes(_onnx(), "m.onnx")["verdict"] == "SAFE_FORMAT"
    assert scan_bytes(_onnx(domain="my.company"), "m.onnx")["verdict"] == "REVIEW"
    assert scan_bytes(_onnx(external="../../etc/passwd"), "m.onnx")["verdict"] == "DANGEROUS"
    assert scan_bytes(b"not protobuf at all", "m.onnx")["verdict"] == "REVIEW"


def test_hdf5_is_reported_as_not_inspected():
    r = scan_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 100, "m.h5")
    assert r["verdict"] == "REVIEW" and "unsupported_format" in _kinds(r)


def test_sandboxed_scan_matches_in_process_scan_and_stays_inert(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    data = pickle.dumps(_Eval())
    sandboxed = scan_sandboxed(data, "evil.pkl")
    assert sandboxed["verdict"] == "DANGEROUS" and sandboxed["sha256"] == scan_bytes(data, "evil.pkl")["sha256"]
    assert FLAG not in os.environ


def test_sandbox_failure_is_treated_as_dangerous(monkeypatch):
    monkeypatch.setattr("app.scanner.model_scan.sys.executable", "/nonexistent/python")
    assert scan_sandboxed(b"x", "m.pkl")["verdict"] == "DANGEROUS"


def test_model_endpoint_scans_without_loading_and_reports_the_verdict(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.security import create_access_token
    from app.main import app

    monkeypatch.delenv(FLAG, raising=False)
    headers = {"Authorization": f"Bearer {create_access_token({'id': 'u7', 'name': 'u7', 'role': 'user'})}"}
    c = TestClient(app)
    evil = c.post("/api/v1/analyze/model", headers=headers, files={"model_file": ("evil.pkl", pickle.dumps(_Eval()), "application/octet-stream")})
    assert evil.status_code == 200 and evil.json()["verdict"] == "DANGEROUS" and FLAG not in os.environ

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    ok = c.post("/api/v1/analyze/model", headers=headers, files={"model_file": ("m.pkl", pickle.dumps(model), "application/octet-stream")})
    body = ok.json()
    assert body["verdict"] == "NO_CODE_EXECUTION_FOUND" and body["model_type"] == "LogisticRegression"
    assert c.post("/api/v1/analyze/model", headers=headers, files={"model_file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400


def test_payload_followed_by_garbage_is_still_reported():
    """pickle.load would run the REDUCE before hitting the garbage, so the scanner must report it too."""
    r = scan_bytes(pickle.dumps(_System())[:-1] + b"\xff\xfe garbage", "m.pkl")
    assert r["verdict"] == "DANGEROUS" and "malformed_pickle" in _kinds(r)


def test_joblib_files_with_raw_array_bytes_parse_cleanly():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.RandomState(0)
    model = RandomForestClassifier(n_estimators=3).fit(rng.rand(60, 4), rng.rand(60) > 0.5)
    for compress in (0, 3, ("gzip", 3), ("bz2", 3), ("lzma", 3)):
        buf = io.BytesIO()
        joblib.dump(model, buf, compress=compress)
        r = scan_bytes(buf.getvalue(), "m.joblib")
        assert r["verdict"] == "NO_CODE_EXECUTION_FOUND", (compress, r["findings"])


def test_payload_hidden_after_a_joblib_array_is_found():
    import joblib
    buf = io.BytesIO()
    joblib.dump({"a": np.arange(10), "evil": _System()}, buf)
    assert scan_bytes(buf.getvalue(), "m.joblib")["verdict"] == "DANGEROUS"


def test_library_gadgets_are_flagged_not_waved_through():
    from research.scanner_bench.build_corpus import proto0
    for module, name in (("pandas.core.computation.eval", "eval"), ("sklearn.utils._testing", "assert_run_python_script_without_output"),
                         ("pydoc", "pipepager")):
        assert scan_bytes(proto0(module, name, ["x"]), "m.pkl")["verdict"] == "DANGEROUS", (module, name)
