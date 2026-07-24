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

import hashlib
import os
import random

from .array import FieldArray
from .compiler import compile_raw_module
from .field import num_limbs
from .jit import jit
from .nttgen import generate_mul_module, generate_ntt_module

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

    def __init__(self, field: type, logn: int, *, w: int | None = None):
        if _np is None:
            raise RuntimeError("the NTT requires numpy for buffer reshaping")
        p = field.modulus
        self.field = field
        self.logn = logn
        self.n = 1 << logn
        self.nl = num_limbs(p)

        if w is not None:
            if pow(w, self.n, p) != 1 or (logn > 0 and pow(w, self.n >> 1, p) == 1):
                raise ValueError(f"{w} is not a primitive 2^{logn}-th root")
            self.w = w
        else:
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
        self._native_tw_fwd = self._packed_twiddles(self.w)
        self._native_tw_inv = self._packed_twiddles(self.winv)
        self._native_transforms = {}
        self._native_mul = None

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

    def _packed_twiddles(self, w: int) -> FieldArray:
        """Twiddle powers for each stage, concatenated without block tiling."""
        p = self.field.modulus
        values = []
        for stage in range(1, self.logn + 1):
            block = 1 << stage
            half = block >> 1
            step = pow(w, self.n // block, p)
            value = 1
            for _ in range(half):
                values.append(value)
                value = value * step % p
        # Size-one transforms never dereference this pointer, but a nonempty
        # allocation keeps its raw ABI address unambiguous.
        return FieldArray(self.field, values or [1])

    def _empty_array(self) -> FieldArray:
        return FieldArray(self.field, _np.zeros((self.n, self.nl), dtype=_np.uint64))

    def _transform_ref(self, fa: FieldArray, tws) -> FieldArray:
        """Original staged implementation, retained as fallback and oracle."""
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
            A = FieldArray(
                self.field, _np.ascontiguousarray(v[:, :half, :]).reshape(-1, nl)
            )
            B = FieldArray(
                self.field, _np.ascontiguousarray(v[:, half:, :]).reshape(-1, nl)
            )
            U, V = _butterfly.map(A, B, tws[st - 1])
            v[:, :half, :] = U._buf.reshape(n // blk, half, nl)
            v[:, half:, :] = V._buf.reshape(n // blk, half, nl)
        return FieldArray(self.field, buf)

    @staticmethod
    def _native_mode() -> str:
        return os.environ.get("FFJIT_NATIVE_NTT", "1").strip().lower()

    def _native_name(self, operation: str) -> str:
        identity = f"{self.field.modulus}:{self.logn}:{operation}"
        digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
        return f"ff_ntt_{operation}_{digest}"

    def _native_transform(self, inverse: bool):
        key = "inverse" if inverse else "forward"
        cached = self._native_transforms.get(key)
        if cached is None:
            module = generate_ntt_module(
                self._native_name(key),
                self.field.modulus,
                self.logn,
                inverse=inverse,
            )
            cached = compile_raw_module(module)
            self._native_transforms[key] = cached
        return cached

    def _transform(self, fa: FieldArray, *, inverse: bool) -> FieldArray:
        if fa.N != self.n or fa.field is not self.field:
            raise ValueError("input does not match this NTT plan")
        tws = self.tw_inv if inverse else self.tw_fwd
        mode = self._native_mode()
        if mode in ("0", "false", "no", "off"):
            out = self._transform_ref(fa, tws)
            return _pointwise_mul.map(out, self.ninv_arr) if inverse else out

        native_tw = self._native_tw_inv if inverse else self._native_tw_fwd
        try:
            kernel = self._native_transform(inverse)
            out = self._empty_array()
            kernel(
                [
                    out.buffer_address(),
                    fa.buffer_address(),
                    native_tw.buffer_address(),
                ]
            )
            return out
        except Exception:
            if mode == "strict":
                raise
            out = self._transform_ref(fa, tws)
            return _pointwise_mul.map(out, self.ninv_arr) if inverse else out

    def ntt(self, fa: FieldArray) -> FieldArray:
        """Forward transform: X[k] = sum_j x[j] * w^(jk), natural order."""
        return self._transform(fa, inverse=False)

    def intt(self, fa: FieldArray) -> FieldArray:
        """Inverse transform: x = n^-1 * NTT_{w^-1}(X)."""
        return self._transform(fa, inverse=True)

    def mul(self, a: FieldArray, b: FieldArray) -> FieldArray:
        """One-call cyclic convolution, with a staged reference fallback."""
        if (
            a.N != self.n
            or b.N != self.n
            or a.field is not self.field
            or b.field is not self.field
        ):
            raise ValueError("operands do not match this NTT plan")
        mode = self._native_mode()
        if mode not in ("0", "false", "no", "off"):
            try:
                if self._native_mul is None:
                    module = generate_mul_module(
                        self._native_name("mul"),
                        self.field.modulus,
                        self.logn,
                        negacyclic=False,
                    )
                    self._native_mul = compile_raw_module(module)
                out = self._empty_array()
                scratch_a = self._empty_array()
                scratch_b = self._empty_array()
                self._native_mul(
                    [
                        out.buffer_address(),
                        a.buffer_address(),
                        b.buffer_address(),
                        scratch_a.buffer_address(),
                        scratch_b.buffer_address(),
                        self._native_tw_fwd.buffer_address(),
                        self._native_tw_inv.buffer_address(),
                    ]
                )
                return out
            except Exception:
                if mode == "strict":
                    raise
        left = self._transform_ref(a, self.tw_fwd)
        right = self._transform_ref(b, self.tw_fwd)
        product = _pointwise_mul.map(left, right)
        out = self._transform_ref(product, self.tw_inv)
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


# ---------------------------------------------------------------------------
# Negacyclic convolution (multiplication in GF(p)[x] / (x^n + 1))
# ---------------------------------------------------------------------------


class NegacyclicPlan:
    """Multiplication in the ring GF(p)[x] / (x^n + 1) via psi-twisting.

    The cyclic NTT computes convolution mod x^n - 1. For the *negacyclic*
    ring (wrapped coefficients pick up a minus sign -- the ring underlying
    Ring-LWE cryptosystems such as Kyber and Dilithium), evaluate at odd
    powers of a primitive 2n-th root psi instead: with a'_i = psi^i * a_i,

        NEGACONV(a, b)_k = psi^{-k} * CYCLICCONV(a', b')_k,

    because psi^n = -1 turns the wraparound x^n = 1 into x^n = -1. Needs
    2-adicity >= logn + 1. The twists are batched pointwise kernel calls,
    so the whole product is still O(n log n) with no per-element Python.
    """

    def __init__(self, field: type, logn: int):
        p = field.modulus
        self.field = field
        self.n = 1 << logn
        psi = root_of_unity(p, logn + 1)
        # Base the cyclic transform on w = psi^2 so the twist and the
        # transform share one coherent root system.
        self.plan = NTTPlan(field, logn, w=psi * psi % p)

        psi_inv = pow(psi, -1, p)
        pows, ipows = [1] * self.n, [1] * self.n
        for i in range(1, self.n):
            pows[i] = pows[i - 1] * psi % p
            ipows[i] = ipows[i - 1] * psi_inv % p
        self.psi_pows = FieldArray(field, pows)
        self.psi_inv_pows = FieldArray(field, ipows)
        self._native_mul = None

    def mul(self, a: FieldArray, b: FieldArray) -> FieldArray:
        if (
            a.N != self.n
            or b.N != self.n
            or a.field is not self.field
            or b.field is not self.field
        ):
            raise ValueError("operands do not match this negacyclic plan")
        mode = self.plan._native_mode()
        if mode not in ("0", "false", "no", "off"):
            try:
                if self._native_mul is None:
                    module = generate_mul_module(
                        self.plan._native_name("negacyclic"),
                        self.field.modulus,
                        self.plan.logn,
                        negacyclic=True,
                    )
                    self._native_mul = compile_raw_module(module)
                out = self.plan._empty_array()
                scratch_a = self.plan._empty_array()
                scratch_b = self.plan._empty_array()
                self._native_mul(
                    [
                        out.buffer_address(),
                        a.buffer_address(),
                        b.buffer_address(),
                        scratch_a.buffer_address(),
                        scratch_b.buffer_address(),
                        self.plan._native_tw_fwd.buffer_address(),
                        self.plan._native_tw_inv.buffer_address(),
                        self.psi_pows.buffer_address(),
                        self.psi_inv_pows.buffer_address(),
                    ]
                )
                return out
            except Exception:
                if mode == "strict":
                    raise
        at = _pointwise_mul.map(a, self.psi_pows)
        bt = _pointwise_mul.map(b, self.psi_pows)
        left = self.plan._transform_ref(at, self.plan.tw_fwd)
        right = self.plan._transform_ref(bt, self.plan.tw_fwd)
        product = _pointwise_mul.map(left, right)
        inverse = self.plan._transform_ref(product, self.plan.tw_inv)
        scaled = _pointwise_mul.map(inverse, self.plan.ninv_arr)
        return _pointwise_mul.map(scaled, self.psi_inv_pows)


_neg_plans = {}


def negacyclic_mul(a: FieldArray, b: FieldArray) -> FieldArray:
    """Product of two length-n coefficient vectors in GF(p)[x] / (x^n + 1)."""
    n = a.N
    if n != b.N or a.field is not b.field:
        raise ValueError("operands must share length and field")
    if n & (n - 1):
        raise ValueError("negacyclic length must be a power of two")
    key = (a.field.modulus, n)
    if key not in _neg_plans:
        _neg_plans[key] = NegacyclicPlan(a.field, n.bit_length() - 1)
    return _neg_plans[key].mul(a, b)
