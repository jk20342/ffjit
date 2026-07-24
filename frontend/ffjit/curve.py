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

Multi-scalar multiplication  sum_i k_i * P_i  uses Pippenger's bucket method
with the bucket accumulation executed as *batched* tree reduction: each round
pairs up pending points across all buckets and performs every addition in one
call into the compiled batch kernel.
"""

from __future__ import annotations

import functools
import math
from typing import List, Sequence, Tuple

from .array import FieldArray
from .field import GF
from .jit import jit


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

    def __repr__(self):
        if self.is_infinity:
            return f"Point(inf, {self.curve.name})"
        x, y = self.to_affine()
        return f"Point({x}, {y}, {self.curve.name})"


# ---------------------------------------------------------------------------
# Pippenger multi-scalar multiplication
# ---------------------------------------------------------------------------

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


def _default_window(n: int) -> int:
    """Window size for Pippenger.

    The classical optimum c ~ log2(n) assumes every group addition costs the
    same. Here the ~2^c sequential bucket-aggregation adds per window pay
    full per-call overhead while the ~n tree-reduction adds are batched
    (near-zero overhead each), which shifts the optimum down to roughly
    log2(n)/2 -- confirmed by measurement (at n=512, c=5 is ~2.8x faster
    than c=9).
    """
    return max(2, min(10, n.bit_length() // 2))


def msm(points: Sequence[Point], scalars: Sequence[int], *,
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
    glv = curve._glv

    pairs = []
    for P, k in zip(points, scalars):
        k = int(k)
        if glv is not None:
            # GLV: one (P, k) becomes (P, k1) and (phi(P), k2) with
            # half-length scalars, halving the number of window passes.
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
    if not pairs:
        return curve.infinity()

    maxbits = max(k.bit_length() for _, k in pairs)
    c = window or _default_window(len(pairs))
    nwin = (maxbits + c - 1) // c
    mask = (1 << c) - 1

    result = curve.infinity()
    for w in range(nwin - 1, -1, -1):
        if not result.is_infinity:
            for _ in range(c):
                result = result.double()

        buckets: dict = {}
        for P, k in pairs:
            d = (k >> (w * c)) & mask
            if d:
                buckets.setdefault(d, []).append(P)
        if not buckets:
            continue
        reduced = _reduce_buckets(curve, buckets)

        # sum_d d * B_d, digits descending: keep a running sum of buckets
        # and weight it by the gap to the next (smaller) digit present.
        digits = sorted(reduced, reverse=True)
        running = curve.infinity()
        acc = curve.infinity()
        for i, d in enumerate(digits):
            running = running + reduced[d]
            gap = d - (digits[i + 1] if i + 1 < len(digits) else 0)
            acc = acc + (gap * running if gap > 1 else running)
        result = result + acc

    return result


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
