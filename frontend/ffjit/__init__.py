"""ffjit -- a JIT compiler for prime-field arithmetic with arbitrary moduli.

Example
-------
>>> import ffjit as ff
>>> F = ff.GF(65537)
>>> @ff.jit
... def f(x, y):
...     return (x * y + x).inv()
>>> f(F(3), F(5))
GF(65537)(...)
"""

from .array import FieldArray
from .curve import Curve, Point, bn254_g1, msm, secp256k1
from .errors import CompileError, TraceError
from .field import GF, FieldVal, num_limbs, storage_bits
from .jit import JittedFunction, jit
from .ntt import intt, ntt, root_of_unity, two_adicity
from .poly import Poly

__all__ = [
    "GF",
    "FieldVal",
    "FieldArray",
    "Poly",
    "Curve",
    "Point",
    "msm",
    "bn254_g1",
    "secp256k1",
    "jit",
    "JittedFunction",
    "TraceError",
    "CompileError",
    "ntt",
    "intt",
    "two_adicity",
    "root_of_unity",
    "storage_bits",
    "num_limbs",
]
__version__ = "0.1.0"
