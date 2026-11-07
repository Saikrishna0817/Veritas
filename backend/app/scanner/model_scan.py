"""Static scanner for model files: "can loading this file run code, and what is in it?"

Nothing here ever deserialises or executes the file. Pickles are walked opcode by opcode with
``pickletools.genops`` and every function or class they would import is checked against a deny list and a
known-safe list. Safe formats (safetensors, GGUF, ONNX, plain ``.npy``) are only structurally validated.

This module uses the standard library only, so it can run in a bare subprocess (``scan_sandboxed``) with
resource limits. A real deployment should also run that worker in its own container without network access;
see docs/roadmap-to-launch.md, Phase 2 item 7.

Verdicts, most to least severe:
  DANGEROUS                 imports something that can run code or touch the system (or could not be resolved)
  REVIEW                    unknown imports, unsupported or unparseable content: a person should look
  NO_CODE_EXECUTION_FOUND   a pickle whose imports are all known data containers (not a guarantee of safety)
  SAFE_FORMAT               a format that cannot contain code (tensors/weights only), structure checks passed
"""
from __future__ import annotations

import ast
import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import pickletools
import struct
import subprocess
import sys
import tempfile
import zlib

MAX_DECOMPRESSED = 400 * 1024 * 1024
MAX_ZIP_MEMBER = 400 * 1024 * 1024
MAX_ZIP_TOTAL = 2 * 1024 * 1024 * 1024
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
SEVERITY_ORDER = {"critical": 3, "high": 3, "medium": 2, "low": 1, "info": 0}

# ── What a pickle may import ────────────────────────────────────────────────────────────────────
DANGEROUS_ROOTS = {
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "pty", "runpy", "importlib", "imp", "ctypes",
    "webbrowser", "code", "codeop", "pickle", "_pickle", "marshal", "shelve", "dill", "cloudpickle", "tempfile",
    "http", "urllib", "urllib2", "requests", "ftplib", "smtplib", "telnetlib", "asyncio", "multiprocessing",
    "threading", "signal", "pdb", "pip", "zipfile", "tarfile", "commands", "popen2", "platform",
    "pathlib", "glob", "fileinput", "linecache", "timeit", "trace", "cProfile", "profile", "bdb",
    "sched", "ssl", "select", "mmap", "resource", "gc", "inspect", "types", "operator",
    "pydoc", "lib2to3", "distutils", "setuptools", "sysconfig", "site", "cgi", "cgitb", "wsgiref", "xmlrpc", "poplib",
    "imaplib", "nntplib", "uuid", "sqlite3", "dbm", "ensurepip", "venv", "idlelib", "doctest", "pstats", "pkgutil",
    "modulefinder", "zipimport", "getpass", "curses", "pexpect", "paramiko", "fabric", "invoke",
    "torch.hub", "torch.jit", "torch.utils", "torch.distributed", "torch.multiprocessing", "torch.testing", "numpy.testing",
    "numpy.lib.npyio", "numpy.lib._npyio_impl", "numpy.ctypeslib", "sklearn.utils._testing", "sklearn.datasets",
    "scipy._lib._testutils", "scipy.io", "pandas.core.computation", "pandas.io", "pandas.util", "pandas.testing",
    "numpy.f2py", "numpy.distutils", "joblib.parallel", "keras.src.utils", "tensorflow.python.ops.gen_io_ops",
}
DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "breakpoint", "memoryview", "classmethod", "staticmethod", "property", "super", "apply", "execfile",
    "reload", "file", "exit", "quit", "help", "id",
}
SAFE_BUILTINS = {
    "set", "frozenset", "slice", "dict", "list", "tuple", "int", "float", "complex", "bool", "str", "bytes",
    "bytearray", "object", "range", "Ellipsis", "NotImplemented", "type",
}
SAFE_PAIRS = {
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("collections", "deque"), ("collections", "Counter"),
    ("copyreg", "_reconstructor"), ("copy_reg", "_reconstructor"), ("datetime", "datetime"), ("datetime", "date"),
    ("datetime", "timedelta"), ("decimal", "Decimal"), ("fractions", "Fraction"),
    ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy", "float64"), ("numpy", "float32"), ("numpy", "int64"),
    ("numpy", "int32"), ("numpy", "bool_"), ("numpy", "uint8"), ("numpy", "matrix"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._mt19937", "MT19937"), ("numpy.random.bit_generator", "SeedSequence"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
    ("_codecs", "encode"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"), ("joblib.numpy_pickle", "NDArrayWrapper"),
    ("joblib.numpy_pickle", "ZNDArrayWrapper"),
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_qtensor"), ("torch", "Size"), ("torch", "device"), ("torch", "Tensor"),
    ("torch.nn.parameter", "Parameter"), ("torch.storage", "_load_from_bytes"),
}
SAFE_PREFIXES = (
    "sklearn.", "scipy.sparse", "scipy.spatial", "scipy.special", "scipy.stats._distn_infrastructure", "xgboost.", "lightgbm.",
    "pandas.core.series", "pandas.core.frame", "pandas.core.indexes", "pandas.core.internals", "pandas.core.arrays",
    "pandas.core.generic", "pandas._libs", "numpy.dtypes", "numpy.ma.core", "numpy.random", "numpy.core.records",
    "torch.nn.modules.", "torch.nn.functional", "torch.optim.", "_loss",
)
SAFE_TORCH_STORAGE = ("FloatStorage", "DoubleStorage", "HalfStorage", "BFloat16Storage", "LongStorage", "IntStorage",
                      "ShortStorage", "CharStorage", "ByteStorage", "BoolStorage", "UntypedStorage", "ComplexFloatStorage")

DANGEROUS_PAIRS = {("torch", "load"), ("numpy", "load"), ("numpy", "fromfile"), ("builtins", "__build_class__"),
                   ("pandas", "read_pickle"), ("pandas", "read_csv"), ("pandas", "read_parquet"), ("pandas", "eval")}


def classify_global(module: str, name: str) -> str:
    """'dangerous' | 'safe' | 'unknown' for something a pickle imports."""
    root = module.split(".")[0]
    if (module, name) in DANGEROUS_PAIRS:
        return "dangerous"
    if module in ("builtins", "__builtin__"):
        return "dangerous" if name in DANGEROUS_BUILTINS else "safe" if name in SAFE_BUILTINS else "unknown"
    if root in DANGEROUS_ROOTS or any(module == d or module.startswith(d + ".") for d in DANGEROUS_ROOTS if "." in d):
        return "dangerous"
    if (module, name) in SAFE_PAIRS:
        return "safe"
    if module == "torch" and name in SAFE_TORCH_STORAGE:
        return "safe"
    if any(module.startswith(p) or module == p.rstrip(".") for p in SAFE_PREFIXES) and module.split(".")[0] != "pandas_":
        return "safe"
    return "unknown"


def _finding(severity: str, kind: str, detail: str, **extra) -> dict:
    return {"severity": severity, "kind": kind, "detail": detail, **extra}


# ── Pickle ──────────────────────────────────────────────────────────────────────────────────────

class _Mark:
    pass


_MARK = _Mark()
_OPAQUE = ("x",)
_CONST_OPAQUE = {"NONE", "NEWTRUE", "NEWFALSE", "FLOAT", "BINFLOAT", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
                 "BYTEARRAY8", "EMPTY_SET", "FROZENSET", "NEXT_BUFFER", "BINPERSID_RESULT"}
_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}
_INT_OPS = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}
MAX_OPS = 50_000_000


def _itemsize(typestr: str) -> int | None:
    t = typestr.lstrip("<>=|")
    digits = "".join(c for c in t[1:] if c.isdigit())
    if not t or not digits:
        return None
    n = int(digits)
    return n * 4 if t[0] == "U" else n


def _walk(stream: io.BytesIO, findings: list, imports: list, label: str, depth: int = 0) -> tuple[int, int, Exception | None]:
    """Run one pickle stream on an abstract machine: nothing is imported, called or built. Returns
    (opcodes_executed, call_opcodes, error). Imports and findings are appended as they are seen, so a payload
    that precedes garbage is still reported."""
    stack: list = []
    memo: dict = {}
    ops = calls = 0

    def pop():
        if not stack:
            raise ValueError("stack underflow")
        return stack.pop()

    def popmark():
        items = []
        while True:
            x = pop()
            if x is _MARK:
                return items[::-1]
            items.append(x)

    def top():
        if not stack:
            raise ValueError("stack underflow")
        return stack[-1]

    try:
        for op, arg, pos in pickletools.genops(stream):
            ops += 1
            if ops > MAX_OPS:
                raise ValueError("too many opcodes")
            n = op.name
            if n in ("PROTO", "FRAME", "READONLY_BUFFER"):
                pass
            elif n == "STOP":
                break
            elif n == "MARK":
                stack.append(_MARK)
            elif n == "POP":
                if top() is _MARK:
                    popmark()
                else:
                    pop()
            elif n == "POP_MARK":
                popmark()
            elif n == "DUP":
                stack.append(top())
            elif n in _STR_OPS:
                stack.append(("s", arg if isinstance(arg, str) else arg.decode("latin1")))
            elif n in _INT_OPS:
                stack.append(("i", int(arg)))
            elif n in _CONST_OPAQUE:
                stack.append(_OPAQUE)
            elif n == "EMPTY_LIST":
                stack.append(("l", []))
            elif n == "EMPTY_TUPLE":
                stack.append(("t", []))
            elif n == "EMPTY_DICT":
                stack.append(("d", {}))
            elif n == "LIST":
                stack.append(("l", popmark()))
            elif n == "TUPLE":
                stack.append(("t", popmark()))
            elif n in ("TUPLE1", "TUPLE2", "TUPLE3"):
                k = int(n[-1])
                items = [pop() for _ in range(k)][::-1]
                stack.append(("t", items))
            elif n == "DICT":
                items = popmark()
                stack.append(("d", {i[1]: v for i, v in zip(items[::2], items[1::2]) if i[0] == "s"}))
            elif n == "APPEND":
                v = pop()
                t = top()
                if t[0] == "l":
                    t[1].append(v)
            elif n == "APPENDS":
                items = popmark()
                t = top()
                if t[0] == "l":
                    t[1].extend(items)
            elif n == "SETITEM":
                v, k = pop(), pop()
                t = top()
                if t[0] == "d" and k[0] == "s":
                    t[1][k[1]] = v
            elif n == "SETITEMS":
                items = popmark()
                t = top()
                if t[0] == "d":
                    for k, v in zip(items[::2], items[1::2]):
                        if k[0] == "s":
                            t[1][k[1]] = v
            elif n == "ADDITEMS":
                popmark()
            elif n in ("GLOBAL", "INST"):
                module, _, attr = str(arg).partition(" ")
                imports.append((module, attr))
                if n == "INST":
                    popmark()
                    calls += 1
                    stack.append(("inst", ("g", module, attr)))
                else:
                    stack.append(("g", module, attr))
            elif n == "STACK_GLOBAL":
                name, module = pop(), pop()
                if module[0] == "s" and name[0] == "s":
                    imports.append((module[1], name[1]))
                    stack.append(("g", module[1], name[1]))
                else:
                    findings.append(_finding("high", "unresolved_import", f"{label}: STACK_GLOBAL whose module or name is not a plain string", offset=pos))
                    stack.append(_OPAQUE)
            elif n == "OBJ":
                items = popmark()
                calls += 1
                stack.append(("inst", items[0] if items else _OPAQUE))
            elif n == "NEWOBJ":
                pop()
                cls = pop()
                calls += 1
                stack.append(("inst", cls))
            elif n == "NEWOBJ_EX":
                pop(); pop()
                cls = pop()
                calls += 1
                stack.append(("inst", cls))
            elif n == "REDUCE":
                args, fn = pop(), pop()
                calls += 1
                stack.append(("call", fn, args))
            elif n == "BUILD":
                state = pop()
                obj = top()
                calls += 1
                if obj[0] == "inst" and obj[1][:1] == ("g",) and obj[1][1:] == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
                    _skip_joblib_array(stream, state, findings, imports, label, depth)
            elif n == "BINPERSID":
                pop()
                stack.append(_OPAQUE)
            elif n == "PERSID":
                stack.append(_OPAQUE)
            elif n in ("EXT1", "EXT2", "EXT4"):
                findings.append(_finding("medium", "extension_registry", f"{label}: uses the pickle extension registry, which hides the import name", offset=pos))
                stack.append(_OPAQUE)
            elif n == "MEMOIZE":
                memo[len(memo)] = top()
            elif n in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = top()
            elif n in ("GET", "BINGET", "LONG_BINGET"):
                if arg not in memo:
                    raise ValueError(f"memo index {arg} was never set")
                stack.append(memo[arg])
            else:
                raise ValueError(f"unsupported opcode {n}")
    except Exception as exc:
        return ops, calls, exc
    return ops, calls, None


def _skip_joblib_array(stream, state, findings, imports, label, depth) -> None:
    """joblib writes raw array bytes between pickle opcodes. Skip exactly that many bytes (using the shape and dtype
    the pickle just declared); an object array is a nested pickle, which is walked like any other."""
    st = state[1] if state[0] == "d" else {}
    dtype_node, shape_node, align = st.get("dtype"), st.get("shape"), st.get("numpy_array_alignment_bytes", ("n",))
    typestr = None
    if dtype_node and dtype_node[0] == "call" and dtype_node[2][0] == "t" and dtype_node[2][1] and dtype_node[2][1][0][0] == "s":
        typestr = dtype_node[2][1][0][1]
    if typestr is None or shape_node is None or shape_node[0] != "t":
        raise ValueError("joblib array with a dtype or shape that could not be read")
    if "O" in typestr:
        if depth > 8:
            raise ValueError("nested object arrays too deep")
        _, _, err = _walk(stream, findings, imports, label + " (object array)", depth + 1)
        if err:
            raise err
        return
    size = _itemsize(typestr)
    if size is None or any(d[0] != "i" or d[1] < 0 for d in shape_node[1]):
        raise ValueError(f"joblib array with unreadable dtype {typestr!r}")
    count = 1
    for d in shape_node[1]:
        count *= d[1]
    if align and align[0] == "i":
        pad = stream.read(1)
        if not pad:
            raise ValueError("truncated joblib array")
        stream.seek(stream.tell() + pad[0])
    end = stream.tell() + count * size
    if end > len(stream.getbuffer()):
        raise ValueError("joblib array is longer than the file")
    stream.seek(end)


def scan_pickle(data: bytes, label: str = "pickle") -> dict:
    """Scan every pickle stream in ``data``. Returns {findings, imports, call_ops, streams}."""
    findings: list[dict] = []
    all_imports: list[tuple[str, str]] = []
    calls = streams = 0
    stream = io.BytesIO(data)
    while stream.tell() < len(data):
        start = stream.tell()
        ops, c, err = _walk(stream, findings, all_imports, label)
        calls += c
        if ops == 0 and err is not None:
            findings.append(_finding("low" if streams == 0 else "info", "not_a_pickle" if streams == 0 else "binary_after_pickle",
                                     f"{label}: {'does not start with a valid pickle opcode (raw weights or another format); pickle loaders fail on it immediately' if streams == 0 else f'{len(data) - start} bytes of non-pickle data after stream {streams}'}"))
            break
        streams += 1
        if err is not None:
            findings.append(_finding("medium" if streams == 1 else "info", "malformed_pickle" if streams == 1 else "binary_after_pickle",
                                     f"{label}: opcodes stop parsing after {ops} ops ({type(err).__name__}: {err}); imports seen before that point are still reported"))
            break
    seen: set[tuple[str, str]] = set()
    for module, attr in all_imports:
        if (module, attr) in seen:
            continue
        seen.add((module, attr))
        kind = classify_global(module, attr)
        if kind == "dangerous":
            findings.append(_finding("critical", "dangerous_import", f"{label}: imports {module}.{attr}, which can run code or access the system", module=module, name=attr))
        elif kind == "unknown":
            findings.append(_finding("medium", "unknown_import", f"{label}: imports {module}.{attr}, which is not on the known-safe list; a pickle can call anything it imports", module=module, name=attr))
    if streams > 1 and not any(f["kind"] == "binary_after_pickle" for f in findings):
        findings.append(_finding("low", "multiple_streams", f"{label}: {streams} pickle streams in one file"))
    return {"findings": findings, "imports": sorted(seen), "call_ops": calls, "streams": streams}


# ── Other formats ───────────────────────────────────────────────────────────────────────────────

def scan_safetensors(data: bytes) -> list[dict]:
    n = struct.unpack("<Q", data[:8])[0]
    if n > MAX_SAFETENSORS_HEADER or 8 + n > len(data):
        return [_finding("medium", "bad_safetensors", "header length is larger than the file or the allowed maximum")]
    try:
        header = json.loads(data[8:8 + n])
    except ValueError:
        return [_finding("medium", "bad_safetensors", "header is not valid JSON")]
    body = len(data) - 8 - n
    spans = []
    for key, t in header.items():
        if key == "__metadata__":
            continue
        try:
            start, end = t["data_offsets"]
            if not (0 <= start <= end <= body):
                raise ValueError
            spans.append((start, end))
        except (KeyError, TypeError, ValueError):
            return [_finding("medium", "bad_safetensors", f"tensor {key!r} has invalid data offsets")]
    spans.sort()
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return [_finding("medium", "bad_safetensors", "tensor byte ranges overlap")]
    return []


def scan_gguf(data: bytes) -> list[dict]:
    version = struct.unpack("<I", data[4:8])[0] if len(data) >= 8 else 0
    return [] if 1 <= version <= 3 else [_finding("medium", "bad_gguf", f"unexpected GGUF version {version}")]


def _npy_header(data: bytes) -> tuple[dict, int]:
    major = data[6]
    hlen, off = (struct.unpack("<H", data[8:10])[0], 10) if major == 1 else (struct.unpack("<I", data[8:12])[0], 12)
    header = ast.literal_eval(data[off:off + hlen].decode("latin1"))
    return header, off + hlen


def scan_npy(data: bytes, label: str = "npy") -> dict:
    try:
        header, body = _npy_header(data)
    except Exception as exc:
        return {"findings": [_finding("medium", "bad_npy", f"{label}: header could not be parsed ({exc})")], "imports": [], "safe": False}
    descr = header.get("descr")
    if "O" in str(descr) or (isinstance(descr, str) and descr.lstrip("|<>=").startswith("O")):
        out = scan_pickle(data[body:], label=f"{label} object array")
        out["findings"].append(_finding("medium", "object_array", f"{label}: object-dtype array is stored as a pickle"))
        return {**out, "safe": False}
    return {"findings": [], "imports": [], "safe": True}


def _read_varint(b: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _pb_fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _read_varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _read_varint(b, i)
        elif wire == 1:
            value, i = b[i:i + 8], i + 8
        elif wire == 2:
            n, i = _read_varint(b, i)
            value, i = b[i:i + n], i + n
            if len(value) != n:
                raise ValueError("truncated field")
        elif wire == 5:
            value, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wire}")
        yield field, wire, value


_STANDARD_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml", "ai.onnx.preview.training"}


def scan_onnx(data: bytes) -> list[dict]:
    """ONNX is protobuf, not code. Risks are custom operators (native code the runtime must load) and
    external-data paths that point outside the model's folder."""
    findings: list[dict] = []
    try:
        graph = next((v for f, w, v in _pb_fields(data) if f == 7 and w == 2), None)
        if graph is None:
            return [_finding("medium", "not_onnx", "no graph found; the file does not parse as an ONNX model")]
        domains, bad_paths, n_nodes = set(), set(), 0
        for f, w, v in _pb_fields(graph):
            if f == 1 and w == 2:       # NodeProto
                n_nodes += 1
                for nf, nw, nv in _pb_fields(v):
                    if nf == 7 and nw == 2:
                        domains.add(nv.decode("utf8", "replace"))
            elif f == 5 and w == 2:     # initializer TensorProto
                key = None
                for tf, tw, tv in _pb_fields(v):
                    if tf == 13 and tw == 2:    # external_data entries
                        k = v_ = None
                        for ef, ew, ev in _pb_fields(tv):
                            if ef == 1 and ew == 2: k = ev.decode("utf8", "replace")
                            if ef == 2 and ew == 2: v_ = ev.decode("utf8", "replace")
                        if k == "location" and v_ and (v_.startswith(("/", "\\")) or ".." in v_.replace("\\", "/").split("/")):
                            bad_paths.add(v_)
        if n_nodes == 0:
            findings.append(_finding("low", "empty_graph", "the ONNX graph has no nodes"))
        for d in sorted(domains - _STANDARD_ONNX_DOMAINS):
            sev = "medium"
            findings.append(_finding(sev, "custom_operator_domain", f"uses operators from the custom domain {d!r}; running it needs native code from outside the file", domain=d))
        for p in sorted(bad_paths):
            findings.append(_finding("high", "external_data_path", f"tensor data is read from {p!r}, outside the model's folder", path=p))
    except Exception as exc:
        return [_finding("medium", "not_onnx", f"does not parse as an ONNX model ({type(exc).__name__}: {exc})")]
    return findings


def _scan_zip(data: bytes) -> dict:
    import zipfile
    findings, imports, members, pickles, npy = [], [], [], 0, 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return {"findings": [_finding("medium", "bad_zip", f"zip could not be read: {exc}")], "imports": [], "members": []}
    infos = zf.infolist()
    if sum(i.file_size for i in infos) > MAX_ZIP_TOTAL:
        return {"findings": [_finding("high", "zip_bomb", "uncompressed size is over 2 GB")], "imports": [], "members": []}
    names = [i.filename for i in infos]
    for info in infos:
        name = info.filename
        members.append(name)
        base = name.rsplit("/", 1)[-1].lower()
        if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
            findings.append(_finding("medium", "path_traversal_name", f"member {name!r} has an unsafe path"))
        torchscript_code = "/code/" in "/" + name and base.endswith(".py") and any(m.endswith("constants.pkl") for m in names)
        if torchscript_code:
            findings.append(_finding("info", "torchscript_source", f"{name!r} is TorchScript source, interpreted by torch.jit (not Python), but TorchScript can call any registered operator"))
        elif base.endswith((".so", ".dll", ".dylib", ".exe", ".sh", ".bat", ".py", ".pyc", ".pyd")):
            findings.append(_finding("high", "executable_member", f"archive contains the executable or script file {name!r}"))
        if info.file_size > MAX_ZIP_MEMBER:
            findings.append(_finding("high", "zip_bomb", f"member {name!r} expands to {info.file_size} bytes"))
            continue
        if base.endswith((".pkl", ".pickle")):
            pickles += 1
            result = scan_pickle(zf.read(info), label=name)
            findings += result["findings"]
            imports += result["imports"]
        elif base.endswith(".npy"):
            npy += 1
            result = scan_npy(zf.read(info), label=name)
            findings += result["findings"]
            imports += result["imports"]
        elif base == "config.json" and b"Lambda" in zf.read(info):
            findings.append(_finding("high", "keras_lambda", "Keras config contains a Lambda layer, which embeds executable code"))
    return {"findings": findings, "imports": imports, "members": members[:200], "pickle_members": pickles, "npy_members": npy}


def _decompress(data: bytes) -> tuple[bytes, str | None]:
    """Strip a gzip / bz2 / xz / zlib wrapper, with a size cap (decompression bombs)."""
    candidates = [(b"\x1f\x8b", lambda: zlib.decompressobj(31), "gzip"), (b"BZh", bz2.BZ2Decompressor, "bz2"),
                  (b"\xfd7zXZ", lzma.LZMADecompressor, "xz"), (b"\x5d\x00\x00", lambda: lzma.LZMADecompressor(lzma.FORMAT_ALONE), "lzma")]
    if len(data) > 2 and data[0] == 0x78 and (data[0] * 256 + data[1]) % 31 == 0:
        candidates.append((b"\x78", lambda: zlib.decompressobj(), "zlib"))
    for magic, factory, name in candidates:
        if data.startswith(magic):
            try:
                out = factory().decompress(data, MAX_DECOMPRESSED + 1)
            except (OSError, EOFError, zlib.error, lzma.LZMAError, ValueError):
                continue       # looked like a header but is not a stream: treat as plain bytes
            if len(out) > MAX_DECOMPRESSED:
                raise ValueError(f"{name} stream expands past {MAX_DECOMPRESSED // 1024 // 1024} MB")
            return out, name
    return data, None


# ── Entry point ─────────────────────────────────────────────────────────────────────────────────

def _verdict(findings: list[dict], safe_format: bool) -> str:
    worst = max((SEVERITY_ORDER[f["severity"]] for f in findings), default=0)
    if worst >= 3:
        return "DANGEROUS"
    if worst == 2:
        return "REVIEW"
    return "SAFE_FORMAT" if safe_format else "NO_CODE_EXECUTION_FOUND"


def scan_bytes(data: bytes, filename: str = "") -> dict:
    ext = os.path.splitext(filename.lower())[1]
    result: dict = {"filename": filename, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "format": "unknown", "findings": [], "imports": [], "can_run_code_on_load": None}
    findings, safe_format = result["findings"], False
    try:
        body, compression = _decompress(data)
        if compression:
            result["compression"] = compression
        if body.startswith(b"GGUF"):
            result.update(format="gguf"); findings += scan_gguf(body); safe_format = True
        elif len(body) > 8 and body[8:9] == b"{" and struct.unpack("<Q", body[:8])[0] < MAX_SAFETENSORS_HEADER and ext in ("", ".safetensors"):
            result.update(format="safetensors"); findings += scan_safetensors(body); safe_format = True
        elif body.startswith(b"PK\x03\x04"):
            z = _scan_zip(body)
            result.update(format="pytorch_zip" if any(m.endswith("data.pkl") for m in z["members"]) else "zip",
                          members=z["members"], imports=[list(i) for i in z["imports"]])
            findings += z["findings"]
            if z.get("pickle_members", 0) == 0 and z.get("npy_members", 0) == 0:
                findings.append(_finding("medium", "unrecognized_zip", "zip with no pickle or array inside; its contents were not inspected"))
            safe_format = z.get("pickle_members", 0) == 0 and z.get("npy_members", 0) > 0
        elif body.startswith(b"\x93NUMPY"):
            n = scan_npy(body); result.update(format="npy", imports=[list(i) for i in n["imports"]]); findings += n["findings"]; safe_format = n["safe"]
        elif body.startswith(b"\x89HDF"):
            result.update(format="hdf5")
            findings.append(_finding("medium", "unsupported_format", "HDF5/Keras .h5 files can embed Lambda layers that run code; this scanner does not read HDF5, so the file was not inspected"))
        elif ext == ".onnx":
            result.update(format="onnx"); findings += scan_onnx(body); safe_format = not findings
        else:
            looks_pickle = body[:1] == b"\x80" or body[-1:] == b"." or ext in (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".ckpt")
            if not looks_pickle:
                result.update(format="unknown")
                findings.append(_finding("medium", "unrecognized_format", "could not identify the file format; it was not inspected"))
            else:
                p = scan_pickle(body)
                result.update(format="pickle", imports=[list(i) for i in p["imports"]], call_ops=p["call_ops"], pickle_streams=p["streams"])
                findings += p["findings"]
    except Exception as exc:
        findings.append(_finding("medium", "scan_error", f"scanner could not finish ({type(exc).__name__}: {exc})"))
    result["verdict"] = _verdict(findings, safe_format)
    result["can_run_code_on_load"] = {"DANGEROUS": True, "SAFE_FORMAT": False, "NO_CODE_EXECUTION_FOUND": False}.get(result["verdict"])
    findings.sort(key=lambda f: -SEVERITY_ORDER[f["severity"]])
    return result


# ── Sandboxed worker ────────────────────────────────────────────────────────────────────────────

def _limit_resources() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))   # the worker writes nothing to disk


def scan_sandboxed(data: bytes, filename: str, timeout: float = 45.0) -> dict:
    """Scan in a separate, isolated Python process: no environment, throwaway cwd, memory/CPU/file limits.
    Network isolation and seccomp are the deployment's job (run this inside a no-network container)."""
    with tempfile.TemporaryDirectory(prefix="veritas-scan-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", os.path.abspath(__file__), "--worker", filename],
                input=data, capture_output=True, timeout=timeout, cwd=cwd, env={},
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
            if proc.returncode == 0:
                return json.loads(proc.stdout)
            reason = f"scanner process exited with {proc.returncode}"
        except subprocess.TimeoutExpired:
            reason = f"scanner timed out after {timeout:.0f}s"
        except Exception as exc:
            reason = f"scanner could not run ({type(exc).__name__}: {exc})"
    return {"filename": filename, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "format": "unknown",
            "imports": [], "findings": [_finding("high", "scanner_failed", f"{reason}; a file that breaks the scanner is treated as dangerous")],
            "verdict": "DANGEROUS", "can_run_code_on_load": True}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.stdout.write(json.dumps(scan_bytes(sys.stdin.buffer.read(), sys.argv[2])))
