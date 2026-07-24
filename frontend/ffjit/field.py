"""Prime field types: ``GF(p)`` factory and concrete ``FieldVal`` elements.

The concrete arithmetic here is pure Python (using CPython big integers). It
serves two roles: an ergonomic value type the JIT accepts and returns, and a
correctness oracle the compiled kernels are tested against.
"""

from __future__ import annotations

import functools
from typing import Union


def storage_bits(p: int) -> int:
    """Storage bit width W = 64 * nlimbs used by the compiler for modulus ``p``.

    Must match ``ElementType::getStorageBitWidth`` in the MLIR dialect exactly:
    ``nlimbs = bit_length(p) // 64 + 1`` so that the Montgomery radix
    ``R = 2**W`` strictly exceeds ``p``.
    """
    nlimbs = p.bit_length() // 64 + 1
    return 64 * nlimbs


def num_limbs(p: int) -> int:
    return p.bit_length() // 64 + 1


class FieldVal:
    """An element of GF(p). Instances are created via a ``GF(p)`` class."""

    __slots__ = ("value",)
    modulus: int  # set on the subclass produced by GF()

    def __init__(self, value: Union[int, "FieldVal"]):
        if isinstance(value, FieldVal):
            value = value.value
        self.value = int(value) % self.modulus

    # -- arithmetic (eager reference implementation) --
    def _coerce(self, other) -> "FieldVal":
        if isinstance(other, FieldVal):
            if other.modulus != self.modulus:
                raise TypeError(
                    f"mixing fields GF({self.modulus}) and GF({other.modulus})"
                )
            return other
        if isinstance(other, int):
            return type(self)(other)
        return NotImplemented

    def __add__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else type(self)(self.value + o.value)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else type(self)(self.value - o.value)

    def __rsub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else type(self)(o.value - self.value)

    def __mul__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else type(self)(self.value * o.value)

    __rmul__ = __mul__

    def __neg__(self):
        return type(self)(-self.value)

    def inv(self) -> "FieldVal":
        """Multiplicative inverse; ``inv(0) == 0`` by convention (matches JIT)."""
        if self.value == 0:
            return type(self)(0)
        return type(self)(pow(self.value, -1, self.modulus))

    def __pow__(self, e: int):
        if e < 0:
            return self.inv() ** (-e)
        return type(self)(pow(self.value, e, self.modulus))

    def __truediv__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else self * o.inv()

    # -- protocol --
    def __int__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, FieldVal):
            return self.modulus == other.modulus and self.value == other.value
        if isinstance(other, int):
            return self.value == other % self.modulus
        return NotImplemented

    def __hash__(self):
        return hash((self.modulus, self.value))

    def __repr__(self):
        return f"GF({self.modulus})({self.value})"


@functools.lru_cache(maxsize=None)
def GF(p: int) -> type:
    """Return a ``FieldVal`` subclass for the prime field GF(p).

    Cached so ``GF(p) is GF(p)`` and element types are stable across calls.
    """
    if p < 2:
        raise ValueError("modulus must be a prime >= 2")

    cls = type(
        f"GF{p}",
        (FieldVal,),
        {
            "modulus": p,
            "storage_bits": storage_bits(p),
            "num_limbs": num_limbs(p),
        },
    )
    return cls
