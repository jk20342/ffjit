"""Number-theoretic transform over NTT-friendly prime fields.

A prime ``p`` supports a radix-2 NTT of size ``n = 2^k`` iff ``2^k | p - 1``
(the multiplicative group, being cyclic of order ``p - 1``, then contains an
element of order ``2^k``). Cryptographic fields are engineered for this: the
BN254 scalar field has 2-adicity 28, BLS12-381's has 32, Goldilocks
``2^64 - 2^32 + 1`` has 32.

Implementation: iterative decimation-in-time Cooley-Tukey. The butterflies

    (a, b, w)  ->  (a + w*b, a - w*b)

are executed by a JIT-compiled multi-output batch kernel over native limb
buffers; numpy performs only the data movement (bit-reversal gather and
per-stage strided copies). Python never touches individual elements during a
transform.

Roots of unity are found *without factoring* ``p - 1``: for random ``x``,
``w = x^((p-1)/2^k)`` has order dividing ``2^k``, and order exactly ``2^k``
with probability 1/2 (iff ``x`` is a non-residue at level ``k``), which we
check with one squaring chain. Expected two trials.
"""

from __future__ import annotations

import random

from .array import FieldArray
from .field import num_limbs
from .jit import jit

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def two_adicity(p: int) -> int:
    """Largest s with 2^s | p - 1."""
    m = p - 1
    s = 0
    while m % 2 == 0:
        s += 1
        m //= 2
    return s


def root_of_unity(p: int, k: int, *, seed: int = 0xF1E1D) -> int:
    """A primitive 2^k-th root of unity mod p (deterministic given ``seed``)."""
    s = two_adicity(p)
    if k > s:
        raise ValueError(f"GF({p}) has 2-adicity {s} < requested {k}")
    if k == 0:
        return 1
    t = (p - 1) >> k
    rng = random.Random(seed)
    while True:
        x = rng.randrange(2, p)
        w = pow(x, t, p)
        # order(w) | 2^k; it is exactly 2^k iff w^(2^(k-1)) != 1
        if pow(w, 1 << (k - 1), p) != 1:
            return w


@jit
def _butterfly(a, b, w):
    t = w * b
    return a + t, a - t


@jit
def _pointwise_mul(a, b):
    return a * b


class NTTPlan:
    """Precomputed data for size-2^logn transforms over one field.

    Holds the root of unity, bit-reversal permutation, and per-stage tiled
    twiddle arrays (forward and inverse), all in native limb buffers.
    """

    def __init__(self, field: type, logn: int):
        if _np is None:
            raise RuntimeError("the NTT requires numpy for buffer reshaping")
        p = field.modulus
        self.field = field
        self.logn = logn
        self.n = 1 << logn
        self.nl = num_limbs(p)

        self.w = root_of_unity(p, logn)
        self.winv = pow(self.w, -1, p)
        self.ninv = pow(self.n, -1, p)

        idx = _np.arange(self.n, dtype=_np.int64)
        rev = _np.zeros(self.n, dtype=_np.int64)
        for bit in range(logn):
            rev |= ((idx >> bit) & 1) << (logn - 1 - bit)
        self.rev = rev

        self.tw_fwd = self._stage_twiddles(self.w)
        self.tw_inv = self._stage_twiddles(self.winv)

        one = FieldArray(field, [self.ninv])
        self.ninv_arr = FieldArray(field, _np.tile(one._buf, (self.n, 1)))

    def _stage_twiddles(self, w: int):
        """For stage s (block length 2^s): [1, wl, wl^2, ...] tiled across
        blocks, where wl = w^(n / 2^s) has order 2^s."""
        p = self.field.modulus
        out = []
        for st in range(1, self.logn + 1):
            blk = 1 << st
            half = blk >> 1
            wl = pow(w, self.n // blk, p)
            tws = [1] * half
            for i in range(1, half):
                tws[i] = tws[i - 1] * wl % p
            fa = FieldArray(self.field, tws)
            tiled = _np.tile(fa._buf, (self.n // blk, 1))
            out.append(FieldArray(self.field, tiled))
        return out

    def _transform(self, fa: FieldArray, tws) -> FieldArray:
        if fa.N != self.n or fa.field is not self.field:
            raise ValueError("input does not match this NTT plan")
        n, nl = self.n, self.nl
        buf = _np.ascontiguousarray(
            _np.frombuffer(fa.as_bytes(), dtype=_np.uint64).reshape(n, nl)[self.rev]
        )
        for st in range(1, self.logn + 1):
            blk = 1 << st
            half = blk >> 1
            v = buf.reshape(n // blk, blk, nl)
            A = FieldArray(self.field, _np.ascontiguousarray(v[:, :half, :]).reshape(-1, nl))
            B = FieldArray(self.field, _np.ascontiguousarray(v[:, half:, :]).reshape(-1, nl))
            U, V = _butterfly.map(A, B, tws[st - 1])
            v[:, :half, :] = U._buf.reshape(n // blk, half, nl)
            v[:, half:, :] = V._buf.reshape(n // blk, half, nl)
        return FieldArray(self.field, buf)

    def ntt(self, fa: FieldArray) -> FieldArray:
        """Forward transform: X[k] = sum_j x[j] * w^(jk), natural order."""
        return self._transform(fa, self.tw_fwd)

    def intt(self, fa: FieldArray) -> FieldArray:
        """Inverse transform: x = n^-1 * NTT_{w^-1}(X)."""
        out = self._transform(fa, self.tw_inv)
        return _pointwise_mul.map(out, self.ninv_arr)


_plans = {}


def get_plan(field: type, logn: int) -> NTTPlan:
    key = (field.modulus, logn)
    if key not in _plans:
        _plans[key] = NTTPlan(field, logn)
    return _plans[key]


def ntt(fa: FieldArray) -> FieldArray:
    n = fa.N
    if n & (n - 1):
        raise ValueError("NTT length must be a power of two")
    return get_plan(fa.field, n.bit_length() - 1).ntt(fa)


def intt(fa: FieldArray) -> FieldArray:
    n = fa.N
    if n & (n - 1):
        raise ValueError("NTT length must be a power of two")
    return get_plan(fa.field, n.bit_length() - 1).intt(fa)
