"""The ``@ffjit.jit`` decorator."""

from __future__ import annotations

import ctypes
import functools
from typing import Callable, Union

from .array import FieldArray
from .compiler import compile_module
from .field import FieldVal
from .mlirgen import generate
from .tracer import trace

Result = Union[FieldVal, tuple]
MapResult = Union[FieldArray, tuple]


class JittedFunction:
    def __init__(self, fn: Callable, *, montgomery: bool = True, opt: str = "-O2"):
        self._fn = fn
        self._name = "ff_" + fn.__name__
        self._montgomery = montgomery
        self._opt = opt
        self._cache = {}  # (moduli tuple) -> (CompiledKernel, result_field)
        functools.update_wrapper(self, fn)

    def _specialize(self, fields):
        key = tuple(f.modulus for f in fields)
        if key in self._cache:
            return self._cache[key]
        ctx, inputs, outputs = trace(self._fn, fields)
        mod = generate(self._name, ctx, inputs, outputs)
        kernel = compile_module(mod, montgomery=self._montgomery, opt=self._opt)
        result_fields = [o.field for o in outputs]
        entry = (kernel, result_fields)
        self._cache[key] = entry
        return entry

    def __call__(self, *args) -> Result:
        fields = []
        int_args = []
        for a in args:
            if not isinstance(a, FieldVal):
                raise TypeError(
                    f"{self._name}: arguments must be GF(p) field elements, "
                    f"got {type(a).__name__}"
                )
            fields.append(type(a))
            int_args.append(a.value)
        kernel, result_fields = self._specialize(fields)
        out = kernel(int_args)
        if len(result_fields) == 1:
            return result_fields[0](out)
        return tuple(f(v) for f, v in zip(result_fields, out))

    def map(self, *arrays) -> MapResult:
        """Evaluate the kernel elementwise over equal-length argument arrays.

        Arguments may be ``FieldArray``s (fast path: the compiled loop runs
        directly over their native limb buffers, zero per-element Python cost)
        or plain sequences of ``GF(p)`` elements (converted to ``FieldArray``
        first). Returns a ``FieldArray`` (or a tuple of them for kernels with
        multiple outputs).
        """
        if not arrays:
            raise TypeError("map requires at least one argument array")

        fas = []
        for col in arrays:
            if isinstance(col, FieldArray):
                fas.append(col)
            else:
                col = list(col)
                if not col or not isinstance(col[0], FieldVal):
                    raise TypeError(
                        "map arguments must be FieldArray or non-empty "
                        "sequences of GF(p) elements"
                    )
                fas.append(FieldArray(type(col[0]), col))

        n = fas[0].N
        for fa in fas:
            if fa.N != n:
                raise ValueError("all argument arrays must have equal length")

        fields = [fa.field for fa in fas]
        kernel, result_fields = self._specialize(fields)

        outs = [
            ctypes.create_string_buffer(n * nb) for nb in kernel.ret_nbytes
        ]
        kernel.map_raw(
            n,
            [ctypes.addressof(o) for o in outs],
            [fa.buffer_address() for fa in fas],
        )
        results = tuple(
            FieldArray._from_raw(f, o.raw, n)
            for f, o in zip(result_fields, outs)
        )
        return results[0] if len(results) == 1 else results

    def mlir(self, *fields) -> str:
        """Return the generated `field`-dialect MLIR for the given arg fields."""
        ctx, inputs, outputs = trace(self._fn, fields)
        return generate(self._name, ctx, inputs, outputs).text


def jit(fn=None, *, montgomery: bool = True, opt: str = "-O2"):
    """Decorate a straight-line field function for JIT compilation.

    Usage::

        F = ffjit.GF(p)

        @ffjit.jit
        def f(x, y):
            return (x * y + x).inv()

        f(F(3), F(5))   # -> GF(p) element
    """
    if fn is not None:
        return JittedFunction(fn, montgomery=montgomery, opt=opt)

    def deco(f):
        return JittedFunction(f, montgomery=montgomery, opt=opt)

    return deco
