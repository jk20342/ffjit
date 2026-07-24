"""ffjit-doctor: diagnose the toolchain an ffjit installation depends on.

ffjit compiles kernels at runtime, so a working install needs more than the
Python package: the ``ffc`` MLIR driver, a ``clang`` able to assemble LLVM
IR, a writable kernel cache, and (for batched/NTT work) numpy. This script
checks each dependency in turn, ending with a real end-to-end kernel
compile, and reports what is missing and how to fix it.

Run as ``ffjit-doctor`` (console script) or ``python -m ffjit._doctor``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _report(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    print("ffjit-doctor")
    good = True

    v = sys.version_info
    good &= _report(
        "Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"
    )

    try:
        import numpy

        _report("numpy (batched kernels, NTT)", True, numpy.__version__)
    except ImportError:
        # numpy is optional: scalar kernels work without it.
        _report(
            "numpy (batched kernels, NTT)", True,
            "not installed -- FieldArray/ntt will be unavailable",
        )

    from .compiler import _cache_dir, _find_clang, _find_ffc
    from .errors import CompileError

    try:
        ffc = _find_ffc()
        good &= _report("ffc (MLIR lowering driver)", True, ffc)
    except CompileError as e:
        good &= _report("ffc (MLIR lowering driver)", False, str(e))
        ffc = None

    try:
        clang = _find_clang()
        out = subprocess.run(
            [clang, "--version"], capture_output=True, text=True
        ).stdout.splitlines()
        ver = out[0] if out else ""
        good &= _report("clang (native codegen)", True, f"{clang} ({ver})")
    except CompileError as e:
        good &= _report("clang (native codegen)", False, str(e))

    try:
        cache = _cache_dir()
        with tempfile.NamedTemporaryFile(dir=cache):
            pass
        good &= _report(
            "kernel cache writable", True,
            f"{cache} (override with FFJIT_CACHE)",
        )
    except OSError as e:
        good &= _report("kernel cache writable", False, str(e))

    if good and ffc:
        try:
            from .field import GF
            from .jit import jit

            F = GF(2**61 - 1)

            @jit
            def _probe(x, y):
                return x * y + x

            r = _probe(F(3), F(5))
            expect = (3 * 5 + 3) % (2**61 - 1)
            good &= _report(
                "end-to-end kernel compile", int(r) == expect,
                "compiled, loaded, and evaluated correctly",
            )
        except Exception as e:  # noqa: BLE001 -- report whatever broke
            good &= _report("end-to-end kernel compile", False, repr(e))
    elif not ffc:
        _report(
            "end-to-end kernel compile", False,
            "skipped: no ffc (build with `make mlir`, or set FFJIT_FFC)",
        )
        good = False

    print()
    if good:
        print("All checks passed -- ffjit is ready.")
        return 0
    print("Some checks FAILED -- see above.")
    return 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
