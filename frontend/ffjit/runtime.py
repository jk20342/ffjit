"""Optional ctypes binding to the C ABI runtime (``libff_rt``).

The runtime currently provides a modular inverse via the binary extended
Euclidean algorithm. The JIT lowers `field.inv` with a pure-IR Fermat ladder
instead, so this binding is used as an independent oracle in tests and is the
intended home for future batched/hand-tuned kernels (MSM, NTT).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

from .field import num_limbs


def _find_runtime() -> Optional[str]:
    env = os.environ.get("FFJIT_RUNTIME")
    if env and Path(env).exists():
        return env
    root = Path(__file__).resolve().parents[2]
    for cand in [
        root / "runtime" / "build" / "libff_rt.so",
        root / "runtime" / "build" / "libff_rt_shared.so",
    ]:
        if cand.exists():
            return str(cand)
    return None


class Runtime:
    def __init__(self, path: Optional[str] = None):
        path = path or _find_runtime()
        if path is None:
            raise RuntimeError(
                "libff_rt.so not found; build it with `make runtime` or set "
                "FFJIT_RUNTIME"
            )
        self._lib = ctypes.CDLL(path)
        self._lib.ff_rt_abi_version.restype = ctypes.c_int32
        self._lib.ff_rt_inv.restype = None
        self._lib.ff_rt_inv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]

    @property
    def abi_version(self) -> int:
        return int(self._lib.ff_rt_abi_version())

    def inv(self, a: int, modulus: int) -> int:
        """Modular inverse a^{-1} mod modulus (inv(0)=0), via ext-Euclid in C."""
        n = num_limbs(modulus)
        nb = n * 8
        a_buf = ctypes.create_string_buffer((a % modulus).to_bytes(nb, "little"), nb)
        m_buf = ctypes.create_string_buffer(modulus.to_bytes(nb, "little"), nb)
        out = ctypes.create_string_buffer(nb)
        self._lib.ff_rt_inv(
            ctypes.cast(out, ctypes.c_void_p),
            ctypes.cast(a_buf, ctypes.c_void_p),
            ctypes.cast(m_buf, ctypes.c_void_p),
            n,
        )
        return int.from_bytes(out.raw, "little")


_default: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global _default
    if _default is None:
        _default = Runtime()
    return _default
