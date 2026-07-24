"""Elliptic curves over GF(q) with JIT-compiled Jacobian point arithmetic.

A short Weierstrass curve  y^2 = x^3 + a x + b  over a prime field. Points
are held in Jacobian projective coordinates (X, Y, Z), representing the
affine point (X/Z^2, Y/Z^3); Z = 0 encodes the point at infinity. Jacobian
form makes both doubling and addition *inversion-free* -- each is a fixed,
branch-free sequence of field multiplications and additions, which is exactly
the shape the tracer compiles into a single kernel:

    double:  (X, Y, Z)                    -> (X3, Y3, Z3)     [~9 field muls]
    add:     (X1, Y1, Z1, X2, Y2, Z2)     -> (X3, Y3, Z3)     [~16 field muls]

The formulas are the standard dbl-2009-l / add-2007-bl ones from the
Explicit-Formulas Database (Bernstein-Lange). The kernels are total functions
on coordinate triples; the exceptional cases the formulas cannot express
(operands at infinity, and H = 0 in addition, i.e. P = +/-Q) are detected in
Python -- for addition, Z3 = 2*Z1*Z2*H vanishes exactly on the exceptional
set, so a zero output is the (cheap, rare) signal to take the slow path.

Multi-scalar multiplication  sum_i k_i * P_i  uses Pippenger's bucket method.
The C runtime emits a compact dependency schedule, and Python submits each
independent round as one call into the compiled point batch kernel.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
from typing import List, Sequence, Tuple

from .array import FieldArray
from .compiler import compile_raw_module
from .curvegen import generate_batch_inv_module
from .field import GF
from .jit import jit
from .runtime import (
    NO_POINT_SLOT,
    POINT_ADD,
    POINT_DOUBLE,
    get_runtime,
)


class Curve:
    """y^2 = x^3 + a*x + b over GF(q). Builds the jitted point kernels."""

    def __init__(self, field: type, a: int, b: int, name: str = ""):
        self.field = field
        self.q = field.modulus
        self.a = int(a) % self.q
        self.b = int(b) % self.q
        self.name = name or f"E(GF({self.q}))"

        a_const = self.a

        if a_const == 0:
            @jit
            def dbl(X, Y, Z):
                A = X * X
                B = Y * Y
                C = B * B
                t = X + B
                D = t * t - A - C
                D = D + D
                E = A + A + A
                F = E * E
                X3 = F - (D + D)
                C8 = C + C
                C8 = C8 + C8
                C8 = C8 + C8
                Y3 = E * (D - X3) - C8
                YZ = Y * Z
                Z3 = YZ + YZ
                return X3, Y3, Z3
        else:
            @jit
            def dbl(X, Y, Z):
                A = X * X
                B = Y * Y
                C = B * B
                t = X + B
                D = t * t - A - C
                D = D + D
                ZZ = Z * Z
                E = A + A + A + a_const * (ZZ * ZZ)
                F = E * E
                X3 = F - (D + D)
                C8 = C + C
                C8 = C8 + C8
                C8 = C8 + C8
                Y3 = E * (D - X3) - C8
                YZ = Y * Z
                Z3 = YZ + YZ
                return X3, Y3, Z3

        @jit
        def add(X1, Y1, Z1, X2, Y2, Z2):
            Z1Z1 = Z1 * Z1
            Z2Z2 = Z2 * Z2
            U1 = X1 * Z2Z2
            U2 = X2 * Z1Z1
            S1 = Y1 * Z2 * Z2Z2
            S2 = Y2 * Z1 * Z1Z1
            H = U2 - U1
            HH = H + H
            I = HH * HH
            J = H * I
            r = S2 - S1
            r = r + r
            V = U1 * I
            X3 = r * r - J - (V + V)
            SJ = S1 * J
            Y3 = r * (V - X3) - (SJ + SJ)
            ZS = Z1 + Z2
            Z3 = (ZS * ZS - Z1Z1 - Z2Z2) * H
            return X3, Y3, Z3

        self._dbl = dbl
        self._add = add
        self._glv = None  # set by enable_glv() for j-invariant-0 curves
        self._aff_add_kernel = None
        self._norm_kernel = None

    @property
    def _aff_add(self):
        """Affine addition given a precomputed slope denominator inverse.

        The inversion 1/(x2 - x1) is hoisted out and computed for a whole
        batch at once with Montgomery's shared-inversion trick (in Python),
        so the kernel is just 3 multiplications -- versus ~16 for the full
        Jacobian add.
        """
        if self._aff_add_kernel is None:
            @jit
            def aff_add(x1, y1, x2, y2, dinv):
                lam = (y2 - y1) * dinv
                x3 = lam * lam - x1 - x2
                y3 = lam * (x1 - x3) - y1
                return x3, y3

            self._aff_add_kernel = aff_add
        return self._aff_add_kernel

    @property
    def _norm(self):
        """(X, Y, 1/Z) -> affine (x, y): Jacobian normalization with the
        inversion hoisted out for batching."""
        if self._norm_kernel is None:
            @jit
            def norm(X, Y, zinv):
                zi2 = zinv * zinv
                return X * zi2, Y * zi2 * zinv

            self._norm_kernel = norm
        return self._norm_kernel

    # -- point construction --
    def point(self, x: int, y: int) -> "Point":
        x, y = int(x) % self.q, int(y) % self.q
        if (y * y - (x * x * x + self.a * x + self.b)) % self.q != 0:
            raise ValueError(f"({x}, {y}) is not on {self.name}")
        return Point(self, x, y, 1)

    def infinity(self) -> "Point":
        return Point(self, 1, 1, 0)

    # -- GLV endomorphism --
    def enable_glv(self, order: int, generator: "Point") -> None:
        """Set up the GLV endomorphism for an a=0 curve of prime ``order``.

        For j-invariant-0 curves, phi(x, y) = (beta*x, y) with beta a
        primitive cube root of unity in GF(q) is an endomorphism satisfying
        phi^2 + phi + 1 = 0; on the order-r subgroup it acts as
        multiplication by lambda, a root of x^2 + x + 1 mod r. Scalars then
        decompose as k = k1 + k2*lambda with |ki| ~ sqrt(r) via a
        lattice-reduced basis found by the extended Euclidean algorithm
        (Gallant-Lambert-Vanstone 2001).
        """
        if self.a != 0:
            raise ValueError("GLV via cube roots requires a = 0")
        q, r = self.q, order
        if (q - 1) % 3 or (r - 1) % 3:
            raise ValueError("GLV requires cube roots of unity in both fields")

        def cube_root_of_unity(m: int) -> int:
            for g in range(2, 100):
                w = pow(g, (m - 1) // 3, m)
                if w != 1:
                    return w
            raise RuntimeError("no cube root of unity found")

        beta = cube_root_of_unity(q)
        lam = cube_root_of_unity(r)

        # Pair beta with the lambda that matches phi on the group: phi(G)
        # must equal lambda*G (else it is lambda^2 = -1-lambda mod r).
        gx, gy = generator.to_affine()
        phi_g = Point(self, beta * gx % q, gy, 1)
        if phi_g != generator._plain_mul(lam):
            lam = (r - 1 - lam) % r
            assert phi_g == generator._plain_mul(lam), "GLV pairing failed"

        v1, v2 = _glv_short_basis(r, lam)
        self._glv = {"beta": beta, "lam": lam, "r": r, "v1": v1, "v2": v2}

    def _endo(self, P: "Point") -> "Point":
        """phi(P) = (beta*X, Y, Z) -- one field multiplication."""
        return Point(self, self._glv["beta"] * P.X % self.q, P.Y, P.Z)

    def __repr__(self):
        return self.name


def _round_div(n: int, d: int) -> int:
    """round(n / d) for integers, d > 0 (round half away from zero)."""
    if n >= 0:
        return (2 * n + d) // (2 * d)
    return -((-2 * n + d) // (2 * d))


def _glv_short_basis(r: int, lam: int):
    """Two short, linearly independent vectors of the lattice
    L = {(x, y) : x + y*lam = 0 mod r}, via the extended Euclidean
    algorithm on (r, lam) run until the remainder drops below sqrt(r)."""
    sqrt_r = math.isqrt(r)
    rs = [r, lam]
    ts = [0, 1]
    i = 1
    while rs[i] > sqrt_r:
        qt = rs[i - 1] // rs[i]
        rs.append(rs[i - 1] - qt * rs[i])
        ts.append(ts[i - 1] - qt * ts[i])
        i += 1
    # rs[i] <= sqrt_r < rs[i-1]; each row gives lattice vector (rs, -ts)
    v1 = (rs[i], -ts[i])
    if len(rs) == i + 1:
        qt = rs[i - 1] // rs[i]
        rs.append(rs[i - 1] - qt * rs[i])
        ts.append(ts[i - 1] - qt * ts[i])
    cand_a = (rs[i - 1], -ts[i - 1])
    cand_b = (rs[i + 1], -ts[i + 1])
    v2 = cand_a if (cand_a[0] ** 2 + cand_a[1] ** 2
                    <= cand_b[0] ** 2 + cand_b[1] ** 2) else cand_b
    return v1, v2


def _glv_decompose(k: int, glv: dict):
    """Write k = k1 + k2*lam (mod r) with |k1|, |k2| = O(sqrt(r)).

    (k, 0) is projected onto the short basis (v1, v2) over the rationals,
    rounded to the nearest lattice point, and subtracted."""
    a1, b1 = glv["v1"]
    a2, b2 = glv["v2"]
    det = a1 * b2 - a2 * b1
    if det < 0:
        a2, b2 = -a2, -b2
        det = -det
    c1 = _round_div(b2 * k, det)
    c2 = _round_div(-b1 * k, det)
    k1 = k - c1 * a1 - c2 * a2
    k2 = -c1 * b1 - c2 * b2
    return k1, k2


class Point:
    """A point in Jacobian coordinates. Immutable."""

    __slots__ = ("curve", "X", "Y", "Z")

    def __init__(self, curve: Curve, X: int, Y: int, Z: int):
        self.curve = curve
        self.X, self.Y, self.Z = X, Y, Z

    @property
    def is_infinity(self) -> bool:
        return self.Z == 0

    def to_affine(self) -> Tuple[int, int]:
        if self.is_infinity:
            raise ValueError("the point at infinity has no affine coordinates")
        q = self.curve.q
        zinv = pow(self.Z, -1, q)
        zinv2 = zinv * zinv % q
        return self.X * zinv2 % q, self.Y * zinv2 * zinv % q

    # -- group law --
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point) or self.curve is not other.curve:
            return NotImplemented
        if self.is_infinity or other.is_infinity:
            return self.is_infinity and other.is_infinity
        q = self.curve.q
        z1z1, z2z2 = self.Z * self.Z % q, other.Z * other.Z % q
        if (self.X * z2z2 - other.X * z1z1) % q != 0:
            return False
        return (self.Y * z2z2 * other.Z - other.Y * z1z1 * self.Z) % q == 0

    def __hash__(self):
        if self.is_infinity:
            return hash((id(self.curve), "inf"))
        return hash((id(self.curve),) + self.to_affine())

    def __neg__(self) -> "Point":
        if self.is_infinity:
            return self
        return Point(self.curve, self.X, (-self.Y) % self.curve.q, self.Z)

    def double(self) -> "Point":
        if self.is_infinity:
            return self
        F = self.curve.field
        X3, Y3, Z3 = self.curve._dbl(F(self.X), F(self.Y), F(self.Z))
        # Z3 = 2*Y*Z = 0 iff Y = 0 (a 2-torsion point): result is infinity,
        # which the coordinates already encode.
        return Point(self.curve, int(X3), int(Y3), int(Z3))

    def __add__(self, other: "Point") -> "Point":
        if not isinstance(other, Point) or self.curve is not other.curve:
            return NotImplemented
        if self.is_infinity:
            return other
        if other.is_infinity:
            return self
        F = self.curve.field
        X3, Y3, Z3 = self.curve._add(
            F(self.X), F(self.Y), F(self.Z), F(other.X), F(other.Y), F(other.Z)
        )
        if int(Z3) == 0:
            # Z3 = 2*Z1*Z2*H with Z1, Z2 != 0, so H = 0: same x-coordinate,
            # meaning P = Q (double) or P = -Q (infinity).
            if self == other:
                return self.double()
            return self.curve.infinity()
        return Point(self.curve, int(X3), int(Y3), int(Z3))

    def __sub__(self, other: "Point") -> "Point":
        return self + (-other)

    def _plain_mul(self, k: int) -> "Point":
        """Left-to-right double-and-add (no endomorphism)."""
        if k < 0:
            return (-self)._plain_mul(-k)
        acc = self.curve.infinity()
        if k == 0 or self.is_infinity:
            return acc
        for bit in bin(k)[2:]:
            acc = acc.double()
            if bit == "1":
                acc = acc + self
        return acc

    def _glv_mul(self, k: int) -> "Point":
        """GLV: split k = k1 + k2*lam, then evaluate k1*P + k2*phi(P)
        jointly (Straus-Shamir), halving the number of doublings."""
        curve = self.curve
        glv = curve._glv
        k %= glv["r"]
        k1, k2 = _glv_decompose(k, glv)

        P1 = self if k1 >= 0 else -self
        k1 = abs(k1)
        P2 = curve._endo(self)
        if k2 < 0:
            P2 = -P2
        k2 = abs(k2)

        table = {1: P1, 2: P2, 3: P1 + P2}
        acc = curve.infinity()
        for i in range(max(k1.bit_length(), k2.bit_length()) - 1, -1, -1):
            acc = acc.double()
            idx = ((k1 >> i) & 1) | (((k2 >> i) & 1) << 1)
            if idx:
                acc = acc + table[idx]
        return acc

    def __rmul__(self, k: int) -> "Point":
        if not isinstance(k, int):
            return NotImplemented
        if self.is_infinity or k == 0:
            return self.curve.infinity()
        glv = self.curve._glv
        if glv is not None:
            k %= glv["r"]
            # GLV pays off once k is well past sqrt(r) in size.
            if k.bit_length() > 140:
                return self._glv_mul(k)
        return self._plain_mul(k)

    __mul__ = __rmul__

    def precompute(self, order: int | None = None,
                   window: int = 8) -> "FixedBase":
        """Build a fixed-base comb table for repeated ``k * self``.

        Precomputes d * 2^(j*c) * P for every window j and digit d, so a
        subsequent scalar multiplication is ~ceil(bits/c) additions and
        *zero* doublings (versus ~bits doublings + bits/2 additions for
        double-and-add). ``order`` defaults to the GLV group order if the
        curve has one.
        """
        if order is None:
            if self.curve._glv is None:
                raise ValueError("order is required on curves without GLV")
            order = self.curve._glv["r"]
        return FixedBase(self, order, window)

    def __repr__(self):
        if self.is_infinity:
            return f"Point(inf, {self.curve.name})"
        x, y = self.to_affine()
        return f"Point({x}, {y}, {self.curve.name})"


class FixedBase:
    """Fixed-base comb precomputation (Lim-Lee style, radix 2^c).

    Stores T[j][d] = d * 2^(j*c) * P for j = 0..ceil(b/c)-1 and
    d = 1..2^c-1. Then k*P = sum_j T[j][digit_j(k)]: one table lookup and
    one addition per window -- no doublings at multiply time. The table is
    built with batched compiled additions (one batch call per digit value,
    each covering all windows at once) and stored in affine form via a
    single shared inversion.

    With c = 8 and a 255-bit order this is a 8160-point table built in
    ~255 batch rounds, and each multiplication costs <= 32 additions --
    roughly an order of magnitude fewer kernel calls than double-and-add.
    """

    __slots__ = ("curve", "order", "c", "nwin", "table")

    def __init__(self, P: Point, order: int, window: int = 8):
        if P.is_infinity:
            raise ValueError("cannot precompute the point at infinity")
        if not 1 <= window <= 16:
            raise ValueError("window must be in [1, 16]")
        curve = P.curve
        self.curve = curve
        self.order = order
        self.c = window
        self.nwin = (order.bit_length() + window - 1) // window

        # bases[j] = 2^(j*c) * P via a doubling chain.
        bases = [P]
        for _ in range(self.nwin - 1):
            Q = bases[-1]
            for _ in range(window):
                Q = Q.double()
            bases.append(Q)

        # rows[j][d] = d * bases[j], filled one digit value at a time with
        # a batched add across all windows.
        rows: List[List[Point]] = [[None] * (1 << window)
                                   for _ in range(self.nwin)]
        accs = list(bases)
        for j in range(self.nwin):
            rows[j][1] = bases[j]
        for d in range(2, 1 << window):
            accs = _batch_add(curve, accs, bases)
            for j in range(self.nwin):
                rows[j][d] = accs[j]

        flat = [rows[j][d] for j in range(self.nwin)
                for d in range(1, 1 << window)]
        flat_affine = _batch_normalize(curve, flat)
        self.table: List[List[Affine]] = []
        stride = (1 << window) - 1
        for j in range(self.nwin):
            self.table.append(flat_affine[j * stride:(j + 1) * stride])

    def mul(self, k: int) -> Point:
        k = int(k) % self.order
        if k == 0:
            return self.curve.infinity()
        mode = _native_msm_mode()
        if mode not in _FALSE_MODES:
            try:
                schedule, ops = get_runtime().fixed_base_schedule(
                    k, self.order.bit_length(), self.c, self.nwin
                )
                inputs = [
                    Point(self.curve, x, y, 1)
                    for row in self.table
                    for x, y in row
                ]
                return _execute_schedule(self.curve, inputs, schedule, ops)
            except Exception:
                if mode == "strict":
                    raise
        return self._mul_ref(k)

    def _mul_ref(self, k: int) -> Point:
        """Original Python comb traversal, retained as a fallback oracle."""
        acc = self.curve.infinity()
        mask = (1 << self.c) - 1
        for j in range(self.nwin):
            d = (k >> (j * self.c)) & mask
            if d:
                x, y = self.table[j][d - 1]
                acc = acc + Point(self.curve, x, y, 1)
        return acc

    def __rmul__(self, k: int) -> Point:
        if not isinstance(k, int):
            return NotImplemented
        return self.mul(k)

    __mul__ = __rmul__

    def __repr__(self):
        return (f"FixedBase({self.curve.name}, c={self.c}, "
                f"{self.nwin * ((1 << self.c) - 1)} points)")


# ---------------------------------------------------------------------------
# Pippenger multi-scalar multiplication
# ---------------------------------------------------------------------------

def _batch_add_points(curve: Curve, Ps: List[Point],
                      Qs: List[Point]) -> List[Point]:
    """Pairwise Ps[i] + Qs[i], tolerating points at infinity (those pairs
    resolve trivially in Python; the rest go through one batch call)."""
    out: List[Point] = [None] * len(Ps)  # type: ignore[list-item]
    idx, fP, fQ = [], [], []
    for i, (P, Q) in enumerate(zip(Ps, Qs)):
        if P.is_infinity:
            out[i] = Q
        elif Q.is_infinity:
            out[i] = P
        else:
            idx.append(i)
            fP.append(P)
            fQ.append(Q)
    if idx:
        for i, R in zip(idx, _batch_add(curve, fP, fQ)):
            out[i] = R
    return out


def _batch_add(curve: Curve, Ps: List[Point], Qs: List[Point]) -> List[Point]:
    """Add Ps[i] + Qs[i] for all i in one compiled batch call.

    All inputs must be finite. Exceptional results (Z3 = 0, i.e. P = +/-Q) are
    fixed up per element on the slow path -- rare for generic inputs.
    """
    F = curve.field
    X3, Y3, Z3 = curve._add.map(
        FieldArray(F, [p.X for p in Ps]),
        FieldArray(F, [p.Y for p in Ps]),
        FieldArray(F, [p.Z for p in Ps]),
        FieldArray(F, [q.X for q in Qs]),
        FieldArray(F, [q.Y for q in Qs]),
        FieldArray(F, [q.Z for q in Qs]),
    )
    xs, ys, zs = X3.to_ints(), Y3.to_ints(), Z3.to_ints()
    out = []
    for i in range(len(Ps)):
        if zs[i] == 0:
            out.append(Ps[i].double() if Ps[i] == Qs[i] else curve.infinity())
        else:
            out.append(Point(curve, xs[i], ys[i], zs[i]))
    return out


def _reduce_buckets(curve: Curve, buckets: dict) -> dict:
    """Tree-reduce each bucket's point list to a single point, batching the
    pairwise additions across all buckets (one compiled call per round)."""
    while True:
        Ps: List[Point] = []
        Qs: List[Point] = []
        slots: List[int] = []
        for d, lst in buckets.items():
            lst[:] = [p for p in lst if not p.is_infinity]
            while len(lst) >= 2:
                Ps.append(lst.pop())
                Qs.append(lst.pop())
                slots.append(d)
        if not Ps:
            break
        for d, R in zip(slots, _batch_add(curve, Ps, Qs)):
            buckets[d].append(R)
    return {
        d: (lst[0] if lst else curve.infinity()) for d, lst in buckets.items()
    }


# -- batch-affine arithmetic (Montgomery shared inversion) --

_FALSE_MODES = ("0", "false", "no", "off", "python")
_batch_inv_kernels = {}


def _native_batch_inv_mode() -> str:
    return os.environ.get("FFJIT_NATIVE_BATCH_INV", "1").strip().lower()


def _native_msm_mode() -> str:
    # Schedule-native MSM is currently workload-dependent and is slower than
    # the Python scheduler on the reference BN254/128 benchmark. Keep it
    # opt-in until it wins consistently; "strict" remains useful for testing.
    return os.environ.get("FFJIT_NATIVE_MSM", "python").strip().lower()

def _batch_inv(q: int, xs: List[int]) -> List[int]:
    """Invert every element of ``xs`` (all nonzero) mod q with a single
    modular inversion: prefix products, one pow(-1), back-substitution.
    3(n-1) multiplications + 1 inversion instead of n inversions."""
    n = len(xs)
    prefix = [1] * (n + 1)
    for i, x in enumerate(xs):
        prefix[i + 1] = prefix[i] * x % q
    inv = pow(prefix[n], -1, q)
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = prefix[i] * inv % q
        inv = inv * xs[i] % q
    return out


def _batch_inv_array(values: FieldArray) -> FieldArray:
    """Invert a FieldArray in one generated call, preserving zero entries."""
    n = values.N
    if n == 0:
        return FieldArray(values.field, [])
    mode = _native_batch_inv_mode()
    if mode not in _FALSE_MODES:
        try:
            key = (values.field.modulus, n)
            kernel = _batch_inv_kernels.get(key)
            if kernel is None:
                identity = f"{values.field.modulus}:{n}:batch_inv"
                digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
                module = generate_batch_inv_module(
                    f"ff_batch_inv_{digest}", values.field.modulus, n
                )
                kernel = compile_raw_module(module)
                _batch_inv_kernels[key] = kernel
            out = FieldArray(values.field, [0] * n)
            scratch = FieldArray(values.field, [0] * n)
            kernel(
                [
                    out.buffer_address(),
                    values.buffer_address(),
                    scratch.buffer_address(),
                ]
            )
            return out
        except Exception:
            if mode == "strict":
                raise
    xs = values.to_ints()
    nonzero = [x for x in xs if x]
    inverses = iter(_batch_inv(values.field.modulus, nonzero))
    return FieldArray(values.field, [next(inverses) if x else 0 for x in xs])


Affine = Tuple[int, int]


def _batch_normalize(curve: Curve, points: Sequence[Point]) -> List[Affine]:
    """Jacobian -> affine using one generated shared-inversion call."""
    F = curve.field
    zinvs = _batch_inv_array(FieldArray(F, [P.Z for P in points]))
    xs, ys = curve._norm.map(
        FieldArray(F, [P.X for P in points]),
        FieldArray(F, [P.Y for P in points]),
        zinvs,
    )
    return list(zip(xs.to_ints(), ys.to_ints()))


def _batch_affine_add(curve: Curve, Ps: List[Affine],
                      Qs: List[Affine]) -> List[Affine | None]:
    """Add affine points pairwise: Ps[i] + Qs[i]. Returns affine points, or
    None for a result at infinity.

    The slope denominators x2 - x1 are inverted together (one shared
    inversion), then the compiled kernel finishes with 3 multiplications per
    add. Pairs with x1 = x2 (a doubling or annihilation) are exceptional
    and handled on the Python slow path -- rare for generic inputs.
    """
    q, F = curve.q, curve.field
    n = len(Ps)
    diffs = [(Qs[i][0] - Ps[i][0]) % q for i in range(n)]
    good = [i for i in range(n) if diffs[i]]
    out: List[Affine | None] = [None] * n

    if good:
        dinvs = _batch_inv_array(FieldArray(F, [diffs[i] for i in good]))
        x3s, y3s = curve._aff_add.map(
            FieldArray(F, [Ps[i][0] for i in good]),
            FieldArray(F, [Ps[i][1] for i in good]),
            FieldArray(F, [Qs[i][0] for i in good]),
            FieldArray(F, [Qs[i][1] for i in good]),
            dinvs,
        )
        for i, x3, y3 in zip(good, x3s.to_ints(), y3s.to_ints()):
            out[i] = (x3, y3)

    for i in range(n):
        if diffs[i] == 0:
            if (Ps[i][1] + Qs[i][1]) % q == 0:
                out[i] = None  # P + (-P) = infinity
            else:
                D = Point(curve, Ps[i][0], Ps[i][1], 1).double()
                out[i] = D.to_affine()
    return out


def _reduce_buckets_affine(curve: Curve, buckets: dict) -> dict:
    """Tree-reduce buckets of *affine* points with batch-affine additions."""
    while True:
        Ps: List[Affine] = []
        Qs: List[Affine] = []
        slots: List[int] = []
        for d, lst in buckets.items():
            while len(lst) >= 2:
                Ps.append(lst.pop())
                Qs.append(lst.pop())
                slots.append(d)
        if not Ps:
            break
        for d, R in zip(slots, _batch_affine_add(curve, Ps, Qs)):
            if R is not None:
                buckets[d].append(R)
    return {
        d: (Point(curve, lst[0][0], lst[0][1], 1) if lst
            else curve.infinity())
        for d, lst in buckets.items()
    }


def _default_window(n: int) -> int:
    """Window size for Pippenger.

    The classical optimum c ~ log2(n) assumes every group addition costs
    the same. Here all phases run as batch kernel calls, but the
    aggregation still pays ~2*2^c batch-call *rounds* of fixed overhead,
    which shifts the optimum down to roughly log2(n) - 3 -- confirmed by
    measurement (c=5 at 256 pairs, c=7 at 1024, c=9 at 4096).
    """
    return max(2, min(12, n.bit_length() - 4))


def _prepare_msm_pairs(curve: Curve, points, scalars):
    glv = curve._glv
    pairs = []
    for P, k in zip(points, scalars):
        k = int(k)
        if glv is not None:
            k %= glv["r"]
            if k == 0 or P.is_infinity:
                continue
            k1, k2 = _glv_decompose(k, glv)
            if k1:
                pairs.append((P if k1 > 0 else -P, abs(k1)))
            if k2:
                Q = curve._endo(P)
                pairs.append((Q if k2 > 0 else -Q, abs(k2)))
        else:
            if k < 0:
                P, k = -P, -k
            if k and not P.is_infinity:
                pairs.append((P, k))
    return pairs


def _batch_double_points(curve: Curve, points: List[Point]) -> List[Point]:
    out: List[Point] = [None] * len(points)  # type: ignore[list-item]
    indices = [i for i, point in enumerate(points) if not point.is_infinity]
    if indices:
        F = curve.field
        X3, Y3, Z3 = curve._dbl.map(
            FieldArray(F, [points[i].X for i in indices]),
            FieldArray(F, [points[i].Y for i in indices]),
            FieldArray(F, [points[i].Z for i in indices]),
        )
        for i, x, y, z in zip(
            indices, X3.to_ints(), Y3.to_ints(), Z3.to_ints()
        ):
            out[i] = Point(curve, x, y, z)
    for i, point in enumerate(points):
        if point.is_infinity:
            out[i] = point
    return out


def _execute_schedule(curve: Curve, inputs, schedule, ops) -> Point:
    slots: List[Point | None] = [None] * schedule.slot_count
    slots[:len(inputs)] = inputs
    cursor = 0
    while cursor < len(ops):
        end = cursor + 1
        while end < len(ops) and ops[end].round == ops[cursor].round:
            end += 1
        round_ops = ops[cursor:end]
        adds = [op for op in round_ops if op.kind == POINT_ADD]
        doubles = [op for op in round_ops if op.kind == POINT_DOUBLE]
        if adds:
            results = _batch_add_points(
                curve,
                [slots[op.lhs] for op in adds],
                [slots[op.rhs] for op in adds],
            )
            for op, result in zip(adds, results):
                slots[op.out] = result
        if doubles:
            results = _batch_double_points(
                curve, [slots[op.lhs] for op in doubles]
            )
            for op, result in zip(doubles, results):
                slots[op.out] = result
        cursor = end
    if schedule.result_slot == NO_POINT_SLOT:
        return curve.infinity()
    result = slots[schedule.result_slot]
    if result is None:
        raise RuntimeError("runtime point schedule produced an empty result slot")
    return result


def _msm_ref(points: Sequence[Point], scalars: Sequence[int], *,
             window: int = 0) -> Point:
    """Multi-scalar multiplication  sum_i scalars[i] * points[i].

    Pippenger's algorithm: split scalars into c-bit windows; per window,
    drop each point into the bucket of its digit, tree-reduce the buckets
    with batched compiled additions, then combine buckets with the
    running-sum identity  sum_d d*B_d = sum_j (running_j * gap_j).
    """
    if len(points) != len(scalars):
        raise ValueError("points and scalars must have equal length")
    if not points:
        raise ValueError("msm of zero points is undefined")
    curve = points[0].curve
    pairs = _prepare_msm_pairs(curve, points, scalars)
    if not pairs:
        return curve.infinity()

    # Normalize every input point to affine once, up front, with a single
    # shared inversion. All bucket work then runs in affine coordinates:
    # a batched add costs 3 kernel multiplications (plus ~3 Python mulmods
    # for the shared-inversion bookkeeping) versus ~16 for Jacobian.
    affine = _batch_normalize(curve, [P for P, _ in pairs])
    apairs = list(zip(affine, (k for _, k in pairs)))

    maxbits = max(k.bit_length() for _, k in pairs)
    c = window or _default_window(len(pairs))
    nwin = (maxbits + c - 1) // c
    mask = (1 << c) - 1

    # All windows are processed simultaneously so that every phase runs as
    # wide batch calls instead of per-window sequential kernel calls:
    #   1. bucket every (point, digit) pair across all windows at once,
    #   2. tree-reduce all (window, digit) buckets together (batch-affine),
    #   3. run the running-sum aggregation digit by digit, batched across
    #      windows (2 * 2^c batch rounds total, instead of ~2 * 2^c
    #      sequential adds *per window*),
    #   4. combine the per-window sums with a final doubling chain.
    buckets: dict = {}
    for A, k in apairs:
        w = 0
        while k:
            d = k & mask
            if d:
                buckets.setdefault((w, d), []).append(A)
            k >>= c
            w += 1
    if not buckets:
        return curve.infinity()
    reduced = _reduce_buckets_affine(curve, buckets)

    # Per-window running-sum aggregation, batched across windows:
    # for d = 2^c-1 .. 1:  running_w += B_{w,d};  acc_w += running_w.
    # This yields acc_w = sum_d d * B_{w,d}.
    running = [curve.infinity()] * nwin
    acc = [curve.infinity()] * nwin
    maxdigit = max(d for _, d in reduced)
    for d in range(maxdigit, 0, -1):
        hits = [w for w in range(nwin) if (w, d) in reduced]
        if hits:
            summed = _batch_add_points(
                curve,
                [running[w] for w in hits],
                [reduced[(w, d)] for w in hits],
            )
            for w, R in zip(hits, summed):
                running[w] = R
        acc = _batch_add_points(curve, acc, running)

    result = curve.infinity()
    for w in range(nwin - 1, -1, -1):
        if not result.is_infinity:
            for _ in range(c):
                result = result.double()
        result = result + acc[w]
    return result


def msm(points: Sequence[Point], scalars: Sequence[int], *,
        window: int = 0) -> Point:
    """Multi-scalar multiplication with runtime-native scheduling."""
    if len(points) != len(scalars):
        raise ValueError("points and scalars must have equal length")
    if not points:
        raise ValueError("msm of zero points is undefined")
    curve = points[0].curve
    if any(P.curve is not curve for P in points):
        raise ValueError("all points must belong to the same curve")
    mode = _native_msm_mode()
    if mode not in _FALSE_MODES:
        try:
            pairs = _prepare_msm_pairs(curve, points, scalars)
            if not pairs:
                return curve.infinity()
            ks = [k for _, k in pairs]
            c = window or _default_window(len(pairs))
            schedule, ops = get_runtime().msm_schedule(
                ks, max(k.bit_length() for k in ks), c
            )
            return _execute_schedule(
                curve, [point for point, _ in pairs], schedule, ops
            )
        except Exception:
            if mode == "strict":
                raise
    return _msm_ref(points, scalars, window=window)


# ---------------------------------------------------------------------------
# Well-known curves
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def bn254_g1() -> Tuple[Curve, Point, int]:
    """BN254 (alt_bn128) G1: y^2 = x^3 + 3 over GF(q). Returns
    (curve, generator, group order r). GLV-enabled."""
    q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    curve = Curve(GF(q), 0, 3, name="BN254 G1")
    G = curve.point(1, 2)
    curve.enable_glv(r, G)
    return curve, G, r


@functools.lru_cache(maxsize=None)
def bls12_381_g1() -> Tuple[Curve, Point, int]:
    """BLS12-381 G1: y^2 = x^3 + 4 over a 381-bit prime field (a 7-limb
    modulus -- one limb wider than BN254). Returns (curve, generator,
    subgroup order r). GLV-enabled."""
    q = 0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB
    r = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
    gx = 0x17F1D3A73197D7942695638C4FA9AC0FC3688C4F9774B905A14E3A3F171BAC586C55E83FF97A1AEFFB3AF00ADB22C6BB
    gy = 0x08B3F481E3AAA0F1A09E30ED741D8AE4FCF5E095D5D00AF600DB18CB2C04B3EDD03CC744A2888AE40CAA232946C5E7E1
    curve = Curve(GF(q), 0, 4, name="BLS12-381 G1")
    G = curve.point(gx, gy)
    curve.enable_glv(r, G)
    return curve, G, r


@functools.lru_cache(maxsize=None)
def secp256k1() -> Tuple[Curve, Point, int]:
    """secp256k1 (the Bitcoin curve): y^2 = x^3 + 7. Returns
    (curve, generator, group order n). GLV-enabled."""
    q = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    curve = Curve(GF(q), 0, 7, name="secp256k1")
    G = curve.point(gx, gy)
    curve.enable_glv(n, G)
    return curve, G, n
