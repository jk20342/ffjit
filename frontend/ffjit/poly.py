"""Dense polynomials over GF(p) with NTT-based multiplication.

``Poly * Poly`` uses the radix-2 NTT (O(n log n)) whenever the result size
fits within the field's 2-adic subgroup and is large enough to beat
schoolbook; otherwise it falls back to schoolbook in Python big ints.
"""

from __future__ import annotations

from typing import List, Sequence, Union

from .array import FieldArray
from .field import FieldVal
from .ntt import _pointwise_mul, get_plan, two_adicity

# below this result size, schoolbook beats NTT plan setup + transforms
_NTT_THRESHOLD = 64


class Poly:
    """coeffs[i] is the coefficient of x^i (dense, little-endian)."""

    __slots__ = ("field", "coeffs")

    def __init__(self, field: type, coeffs: Sequence[Union[int, FieldVal]]):
        self.field = field
        p = field.modulus
        cs = [int(c) % p for c in coeffs]
        while len(cs) > 1 and cs[-1] == 0:
            cs.pop()
        if not cs:
            cs = [0]
        self.coeffs = cs

    # -- structure --
    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __len__(self):
        return len(self.coeffs)

    def __eq__(self, other):
        return (
            isinstance(other, Poly)
            and self.field.modulus == other.field.modulus
            and self.coeffs == other.coeffs
        )

    def __repr__(self):
        if len(self.coeffs) <= 8:
            body = ", ".join(map(str, self.coeffs))
        else:
            body = ", ".join(map(str, self.coeffs[:4])) + ", ..."
        return f"Poly(GF({self.field.modulus}), [{body}], deg={self.degree})"

    # -- ring operations --
    def _check(self, other: "Poly"):
        if not isinstance(other, Poly) or other.field.modulus != self.field.modulus:
            raise TypeError("polynomials must be over the same field")

    def __add__(self, other: "Poly") -> "Poly":
        self._check(other)
        p = self.field.modulus
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0] * (n - len(self.coeffs))
        b = other.coeffs + [0] * (n - len(other.coeffs))
        return Poly(self.field, [(x + y) % p for x, y in zip(a, b)])

    def __sub__(self, other: "Poly") -> "Poly":
        self._check(other)
        p = self.field.modulus
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0] * (n - len(self.coeffs))
        b = other.coeffs + [0] * (n - len(other.coeffs))
        return Poly(self.field, [(x - y) % p for x, y in zip(a, b)])

    def __mul__(self, other: "Poly") -> "Poly":
        self._check(other)
        out_len = len(self.coeffs) + len(other.coeffs) - 1
        logn = (out_len - 1).bit_length()  # next power of two >= out_len
        if out_len >= _NTT_THRESHOLD and logn <= two_adicity(self.field.modulus):
            return self._mul_ntt(other, out_len, logn)
        return self._mul_schoolbook(other, out_len)

    def _mul_schoolbook(self, other: "Poly", out_len: int) -> "Poly":
        p = self.field.modulus
        out = [0] * out_len
        for i, a in enumerate(self.coeffs):
            if a == 0:
                continue
            for j, b in enumerate(other.coeffs):
                out[i + j] = (out[i + j] + a * b) % p
        return Poly(self.field, out)

    def _mul_ntt(self, other: "Poly", out_len: int, logn: int) -> "Poly":
        n = 1 << logn
        plan = get_plan(self.field, logn)
        fa = FieldArray(self.field, self.coeffs + [0] * (n - len(self.coeffs)))
        fb = FieldArray(self.field, other.coeffs + [0] * (n - len(other.coeffs)))
        A = plan.ntt(fa)
        B = plan.ntt(fb)
        C = _pointwise_mul.map(A, B)
        c = plan.intt(C)
        return Poly(self.field, c.to_ints()[:out_len])

    # -- evaluation --
    def __call__(self, x: Union[int, FieldVal]) -> FieldVal:
        p = self.field.modulus
        xv = int(x) % p
        acc = 0
        for c in reversed(self.coeffs):
            acc = (acc * xv + c) % p
        return self.field(acc)

    def evaluate_batch(self, xs: FieldArray) -> List[FieldVal]:
        """Evaluate at many points (Python Horner per point; a jitted
        multipoint evaluation is future work)."""
        return [self(v) for v in xs.to_ints()]
