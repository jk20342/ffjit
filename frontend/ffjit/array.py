"""``FieldArray``: a batch of GF(p) elements stored in a native limb buffer.

The data lives as a contiguous ``(N, nlimbs)`` little-endian ``uint64`` buffer,
so JIT-compiled batch kernels operate on it directly (the buffer pointer is
handed to the compiled loop) with no per-element Python marshalling. Converting
to/from Python integers happens only at the boundaries.

Falls back to a ``bytearray`` buffer when numpy is unavailable.
"""

from __future__ import annotations

from typing import Iterable, List

from .field import num_limbs

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


class FieldArray:
    def __init__(self, field: type, data):
        self.field = field
        self.nlimbs = num_limbs(field.modulus)
        self.elem_bytes = self.nlimbs * 8

        if _np is not None and isinstance(data, _np.ndarray):
            assert data.dtype == _np.uint64 and data.ndim == 2
            self._buf = _np.ascontiguousarray(data)
            self.N = data.shape[0]
            self._raw = None
        else:
            vals = [int(v) % field.modulus for v in data]
            self.N = len(vals)
            nb = self.elem_bytes
            raw = bytearray(self.N * nb)
            for i, v in enumerate(vals):
                raw[i * nb:(i + 1) * nb] = v.to_bytes(nb, "little")
            if _np is not None:
                self._buf = (
                    _np.frombuffer(bytes(raw), dtype=_np.uint64)
                    .reshape(self.N, self.nlimbs)
                    .copy()
                )
                self._raw = None
            else:
                self._buf = None
                self._raw = raw

    @classmethod
    def _from_raw(cls, field: type, raw: bytes, n: int) -> "FieldArray":
        obj = cls.__new__(cls)
        obj.field = field
        obj.nlimbs = num_limbs(field.modulus)
        obj.elem_bytes = obj.nlimbs * 8
        obj.N = n
        if _np is not None:
            obj._buf = (
                _np.frombuffer(bytes(raw), dtype=_np.uint64)
                .reshape(n, obj.nlimbs)
                .copy()
            )
            obj._raw = None
        else:
            obj._buf = None
            obj._raw = bytearray(raw)
        return obj

    def buffer_address(self):
        """Address of the contiguous element buffer (for the batch ABI)."""
        import ctypes
        if self._buf is not None:
            return self._buf.ctypes.data
        return ctypes.addressof((ctypes.c_char * len(self._raw)).from_buffer(self._raw))

    def as_bytes(self) -> bytes:
        if self._buf is not None:
            return self._buf.tobytes()
        return bytes(self._raw)

    def to_list(self) -> List:
        raw = self.as_bytes()
        nb = self.elem_bytes
        return [
            self.field(int.from_bytes(raw[i * nb:(i + 1) * nb], "little"))
            for i in range(self.N)
        ]

    def to_ints(self) -> List[int]:
        raw = self.as_bytes()
        nb = self.elem_bytes
        return [
            int.from_bytes(raw[i * nb:(i + 1) * nb], "little") for i in range(self.N)
        ]

    def __len__(self):
        return self.N

    def __repr__(self):
        return f"FieldArray(GF({self.field.modulus}), N={self.N})"
