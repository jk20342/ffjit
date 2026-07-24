"""Optional ctypes binding to the versioned C runtime (``libff_rt``).

The runtime provides an independent modular-inverse oracle and compact MSM
and fixed-base point-operation schedules. Field arithmetic remains in
generated kernels; Python owns point storage and dispatches independent
schedule rounds to those kernels.
"""

from __future__ import annotations

import ctypes
import os
import sysconfig
from pathlib import Path
from typing import Optional

from .field import num_limbs

RUNTIME_ABI_VERSION = 3
MSM_SCHEDULE_VERSION = 1
POINT_ADD = 1
POINT_DOUBLE = 2
NO_POINT_SLOT = (1 << 64) - 1

_RUNTIME_NAMES = (
    "libff_rt.so",
    "libff_rt.dylib",
    "ff_rt.dll",
    "libff_rt_shared.so",
)


def _runtime_in(directory: Path) -> Optional[str]:
    for name in _RUNTIME_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def find_runtime() -> Optional[str]:
    """Return the discovered runtime library path, or None if unavailable."""
    env = os.environ.get("FFJIT_RUNTIME")
    if env:
        configured = Path(env).expanduser()
        if configured.is_dir():
            found = _runtime_in(configured)
            if found:
                return found
        elif configured.is_file():
            return str(configured.resolve())

    root = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    multiarch = sysconfig.get_config_var("MULTIARCH")
    system_directories = [
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
    ]
    if multiarch:
        system_directories.extend(
            [Path("/usr/lib") / multiarch, Path("/lib") / multiarch]
        )
    for directory in (
        package / ".libs",
        root / "runtime" / "build",
        root / "runtime" / "build" / "lib",
        root / "runtime" / "build" / "Release",
        root / "runtime" / "build" / "Debug",
        root / "lib",
        *system_directories,
    ):
        found = _runtime_in(directory)
        if found:
            return found
    return None


_find_runtime = find_runtime


class PointOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("round", ctypes.c_uint32),
        ("lhs", ctypes.c_uint64),
        ("rhs", ctypes.c_uint64),
        ("out", ctypes.c_uint64),
    ]


class PointSchedule(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("window", ctypes.c_uint32),
        ("window_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("input_count", ctypes.c_uint64),
        ("slot_count", ctypes.c_uint64),
        ("op_count", ctypes.c_uint64),
        ("result_slot", ctypes.c_uint64),
    ]


class Runtime:
    def __init__(self, path: Optional[str] = None):
        path = path or find_runtime()
        if path is None:
            raise RuntimeError(
                "libff_rt.so not found; build it with `make runtime` or set "
                "FFJIT_RUNTIME"
            )
        path_obj = Path(path).expanduser()
        self._path = str(path_obj.resolve()) if path_obj.exists() else path
        try:
            self._lib = ctypes.CDLL(self._path)
        except OSError as exc:
            raise RuntimeError(
                f"failed to load ffjit runtime at {self._path!r}: {exc}"
            ) from exc
        try:
            abi_fn = self._lib.ff_rt_abi_version
        except AttributeError as exc:
            raise RuntimeError(
                f"ffjit runtime at {self._path!r} has no ABI version symbol"
            ) from exc
        abi_fn.restype = ctypes.c_int32
        actual_abi = int(abi_fn())
        if actual_abi != RUNTIME_ABI_VERSION:
            raise RuntimeError(
                "ffjit runtime ABI mismatch: "
                f"frontend requires {RUNTIME_ABI_VERSION}, "
                f"but {self._path!r} provides {actual_abi}"
            )
        self._lib.ff_rt_inv.restype = None
        self._lib.ff_rt_inv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        self._lib.ff_rt_msm_schedule.restype = ctypes.c_int32
        self._lib.ff_rt_msm_schedule.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(PointSchedule),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.ff_rt_fixed_base_schedule.restype = ctypes.c_int32
        self._lib.ff_rt_fixed_base_schedule.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(PointSchedule),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

    @property
    def path(self) -> str:
        return self._path

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

    @staticmethod
    def _scalar_buffer(scalars, nlimbs: int):
        nb = nlimbs * 8
        raw = b"".join(int(k).to_bytes(nb, "little") for k in scalars)
        return ctypes.create_string_buffer(raw, len(raw))

    @staticmethod
    def _finish_schedule(fn, args):
        schedule = PointSchedule()
        status = fn(*args, ctypes.byref(schedule), None, 0)
        if status != 0:
            raise RuntimeError(f"runtime schedule query failed with status {status}")
        if schedule.version != MSM_SCHEDULE_VERSION:
            raise RuntimeError(
                f"unsupported point schedule version {schedule.version}"
            )
        ops = (PointOp * schedule.op_count)()
        status = fn(
            *args,
            ctypes.byref(schedule),
            ctypes.cast(ops, ctypes.c_void_p),
            schedule.op_count,
        )
        if status != 0:
            raise RuntimeError(f"runtime schedule build failed with status {status}")
        return schedule, list(ops)

    def msm_schedule(self, scalars, scalar_bits: int, window: int):
        """Return the runtime's compact Pippenger point-operation schedule."""
        scalars = [int(k) for k in scalars]
        if not scalars or any(k < 0 for k in scalars):
            raise ValueError("scheduled MSM scalars must be nonnegative and nonempty")
        nlimbs = max(1, (scalar_bits + 63) // 64)
        buf = self._scalar_buffer(scalars, nlimbs)
        args = (
            ctypes.cast(buf, ctypes.c_void_p),
            len(scalars),
            nlimbs,
            scalar_bits,
            window,
        )
        return self._finish_schedule(self._lib.ff_rt_msm_schedule, args)

    def fixed_base_schedule(
        self, scalar: int, scalar_bits: int, window: int, window_count: int
    ):
        """Return a balanced schedule for one fixed-base comb lookup."""
        scalar = int(scalar)
        if scalar < 0:
            raise ValueError("fixed-base schedule scalar must be nonnegative")
        nlimbs = max(1, (scalar_bits + 63) // 64)
        buf = self._scalar_buffer([scalar], nlimbs)
        args = (
            ctypes.cast(buf, ctypes.c_void_p),
            nlimbs,
            scalar_bits,
            window,
            window_count,
        )
        return self._finish_schedule(self._lib.ff_rt_fixed_base_schedule, args)


_default: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global _default
    if _default is None:
        _default = Runtime()
    return _default
