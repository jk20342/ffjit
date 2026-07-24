"""Backend: lower generated MLIR to a shared object and load it via ctypes.

Flow:  MLIR text --ffc--> LLVM IR  --(+ptr-ABI wrapper)--> clang -> .so -> dlopen

Field elements cross the ABI boundary as little-endian limb buffers (pointers),
which sidesteps platform rules for passing wide integers (``i256`` etc.) by
value and works uniformly for any modulus size.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .errors import CompileError
from .mlirgen import GeneratedModule


def _find_ffc() -> str:
    env = os.environ.get("FFJIT_FFC")
    if env and Path(env).exists():
        return env
    # repo layout: <root>/frontend/ffjit/compiler.py -> <root>/mlir/build/...
    root = Path(__file__).resolve().parents[2]
    cand = root / "mlir" / "build" / "tools" / "ffc" / "ffc"
    if cand.exists():
        return str(cand)
    found = shutil.which("ffc")
    if found:
        return found
    raise CompileError(
        "cannot locate the 'ffc' compiler driver. Build it with `make mlir` "
        "from the repository root, or point FFJIT_FFC at the binary."
    )


def _find_clang() -> str:
    for name in ("clang", "clang-21", "clang-20", "clang-19"):
        p = shutil.which(name)
        if p:
            return p
    raise CompileError(
        "cannot locate clang to assemble the shared object; install clang "
        "(e.g. `apt install clang`) and ensure it is on PATH"
    )


def _cache_dir() -> Path:
    """Kernel cache directory; override with the FFJIT_CACHE env var."""
    d = Path(os.environ.get("FFJIT_CACHE", Path.cwd() / ".ffjit_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd: List[str], what: str) -> None:
    """Run a toolchain command, surfacing its stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise CompileError(
            f"{what} failed (exit {proc.returncode}): {cmd[0]}\n{detail}"
        )


def _ret_type(mod: GeneratedModule) -> str:
    """LLVM return type of the kernel.

    MLIR's func-to-llvm packs multiple results into an anonymous struct.
    """
    if len(mod.ret_bits) == 1:
        return f"i{mod.ret_bits[0]}"
    return "{ " + ", ".join(f"i{w}" for w in mod.ret_bits) + " }"


def _abi_wrapper(mod: GeneratedModule) -> str:
    """LLVM IR for `void <name>_abi(ptr out0, ..., ptr a0, ...)`."""
    nouts = len(mod.ret_bits)
    outs = [f"ptr %out{k}" for k in range(nouts)]
    args = [f"ptr %a{i}" for i in range(len(mod.arg_bits))]
    sig = ", ".join(outs + args)
    rty = _ret_type(mod)

    lines = [f"define void @{mod.name}_abi({sig}) {{"]
    call_args = []
    for i, w in enumerate(mod.arg_bits):
        lines.append(f"  %v{i} = load i{w}, ptr %a{i}, align 8")
        call_args.append(f"i{w} %v{i}")
    lines.append(f"  %r = call {rty} @{mod.name}({', '.join(call_args)})")
    if nouts == 1:
        lines.append(f"  store i{mod.ret_bits[0]} %r, ptr %out0, align 8")
    else:
        for k, w in enumerate(mod.ret_bits):
            lines.append(f"  %r{k} = extractvalue {rty} %r, {k}")
            lines.append(f"  store i{w} %r{k}, ptr %out{k}, align 8")
    lines.append("  ret void")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _batch_wrapper(mod: GeneratedModule) -> str:
    """LLVM IR for a batched loop:

        void <name>_batch(i64 n, ptr out0, ..., ptr a0, ...)

    Each buffer is a contiguous array of ``n`` elements; element ``j`` occupies
    ``bits[j] // 8`` bytes (little-endian). We index by explicit byte offsets
    so the stride exactly matches how the frontend packs the buffers,
    independent of any LLVM integer alloc-size padding. clang inlines the
    scalar kernel into this loop, so compile time is independent of ``n``.
    """
    nargs = len(mod.arg_bits)
    nouts = len(mod.ret_bits)
    out_ptrs = [f"ptr %out{k}" for k in range(nouts)]
    arg_ptrs = [f"ptr %a{i}" for i in range(nargs)]
    sig = ", ".join(["i64 %n"] + out_ptrs + arg_ptrs)
    rty = _ret_type(mod)

    L = [f"define void @{mod.name}_batch({sig}) {{"]
    L.append("entry:")
    L.append("  %pos = icmp sgt i64 %n, 0")
    L.append("  br i1 %pos, label %loop, label %done")
    L.append("loop:")
    L.append("  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]")
    call_args = []
    for j, w in enumerate(mod.arg_bits):
        nb = w // 8
        L.append(f"  %off{j} = mul i64 %i, {nb}")
        L.append(f"  %p{j} = getelementptr i8, ptr %a{j}, i64 %off{j}")
        L.append(f"  %v{j} = load i{w}, ptr %p{j}, align 8")
        call_args.append(f"i{w} %v{j}")
    L.append(f"  %r = call {rty} @{mod.name}({', '.join(call_args)})")
    for k, w in enumerate(mod.ret_bits):
        nb = w // 8
        val = "%r" if nouts == 1 else f"%r{k}"
        if nouts > 1:
            L.append(f"  %r{k} = extractvalue {rty} %r, {k}")
        L.append(f"  %ooff{k} = mul i64 %i, {nb}")
        L.append(f"  %po{k} = getelementptr i8, ptr %out{k}, i64 %ooff{k}")
        L.append(f"  store i{w} {val}, ptr %po{k}, align 8")
    L.append("  %inext = add i64 %i, 1")
    L.append("  %cont = icmp slt i64 %inext, %n")
    L.append("  br i1 %cont, label %loop, label %done")
    L.append("done:")
    L.append("  ret void")
    L.append("}")
    return "\n".join(L) + "\n"


class CompiledKernel:
    def __init__(self, so_path: str, mod: GeneratedModule):
        self.nargs = len(mod.arg_bits)
        self.nouts = len(mod.ret_bits)
        self.arg_nbytes = [w // 8 for w in mod.arg_bits]
        self.ret_nbytes = [w // 8 for w in mod.ret_bits]

        self._lib = ctypes.CDLL(so_path)
        self._fn = getattr(self._lib, f"{mod.name}_abi")
        self._fn.restype = None
        self._fn.argtypes = [ctypes.c_void_p] * (self.nouts + self.nargs)
        self._batch = getattr(self._lib, f"{mod.name}_batch")
        self._batch.restype = None
        self._batch.argtypes = (
            [ctypes.c_size_t]
            + [ctypes.c_void_p] * (self.nouts + self.nargs)
        )

    def __call__(self, arg_ints: List[int]):
        """Returns an int (single output) or a tuple of ints."""
        bufs = [
            ctypes.create_string_buffer(
                (v % (1 << (nb * 8))).to_bytes(nb, "little"), nb
            )
            for v, nb in zip(arg_ints, self.arg_nbytes)
        ]
        outs = [ctypes.create_string_buffer(nb) for nb in self.ret_nbytes]
        self._fn(
            *[ctypes.cast(o, ctypes.c_void_p) for o in outs],
            *[ctypes.cast(b, ctypes.c_void_p) for b in bufs],
        )
        vals = tuple(int.from_bytes(o.raw, "little") for o in outs)
        return vals[0] if self.nouts == 1 else vals

    def map_raw(self, n: int, out_addresses: List[int],
                arg_addresses: List[int]) -> None:
        """Run the batch loop over raw contiguous limb buffers (zero-copy)."""
        self._batch(n, *out_addresses, *arg_addresses)


def compile_module(mod: GeneratedModule, *, montgomery: bool = True,
                   opt: str = "-O2") -> CompiledKernel:
    # abi=3: multi-output struct-return wrappers (outs-first pointer order)
    key_src = mod.text + f"|mont={montgomery}|opt={opt}|abi=3"
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache = _cache_dir()
    so_path = cache / f"{mod.name}_{key}.so"
    if so_path.exists():
        return CompiledKernel(str(so_path), mod)

    # Serialize concurrent builds of the same kernel across processes; the
    # .so appears in the cache atomically (build to temp name, then rename).
    lock_path = cache / f"{mod.name}_{key}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if so_path.exists():  # another process built it while we waited
            return CompiledKernel(str(so_path), mod)

        ffc = _find_ffc()
        clang = _find_clang()
        tmp_so = cache / f"{mod.name}_{key}.so.tmp{os.getpid()}"

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            mlir_path = tdp / "kernel.mlir"
            ll_path = tdp / "kernel.ll"
            mlir_path.write_text(mod.text)

            cmd = [ffc, str(mlir_path), "-o", str(ll_path), "--emit=llvm"]
            if not montgomery:
                cmd.append("--no-montgomery")
            _run(cmd, "MLIR-to-LLVM lowering")

            with open(ll_path, "a") as f:
                f.write("\n")
                f.write(_abi_wrapper(mod))
                f.write("\n")
                f.write(_batch_wrapper(mod))

            try:
                _run(
                    [clang, "-shared", "-fPIC", opt,
                     "-o", str(tmp_so), str(ll_path)],
                    "native code generation",
                )
                os.replace(tmp_so, so_path)
            finally:
                tmp_so.unlink(missing_ok=True)

    return CompiledKernel(str(so_path), mod)
