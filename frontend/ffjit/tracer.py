"""A tiny operator-overloading tracer.

Calling a decorated function with ``Tracer`` arguments records the field
operations into a straight-line DAG (no Python control flow is captured; the
target domain -- cryptographic kernels -- is overwhelmingly branch-free). The
DAG is then handed to :mod:`ffjit.mlirgen`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .errors import TraceError
from .field import FieldVal


@dataclass
class Node:
    op: str  # input | const | add | sub | mul | neg | inv | pow
    field: type                  # the GF(p) subclass this value lives in
    args: Tuple["Node", ...] = ()
    const_value: Optional[int] = None
    exponent: Optional[int] = None
    index: Optional[int] = None  # for inputs: positional arg index
    id: int = -1


class TraceContext:
    """Collects nodes for one trace."""

    def __init__(self):
        self.nodes: List[Node] = []

    def add(self, node: Node) -> Node:
        node.id = len(self.nodes)
        self.nodes.append(node)
        return node


class Tracer:
    """A symbolic field element flowing through a traced function."""

    __slots__ = ("node", "_ctx")

    def __init__(self, node: Node, ctx: TraceContext):
        self.node = node
        self._ctx = ctx

    @property
    def field(self) -> type:
        return self.node.field

    def _wrap(self, node: Node) -> "Tracer":
        return Tracer(self._ctx.add(node), self._ctx)

    def _lift(self, other) -> Optional["Tracer"]:
        if isinstance(other, Tracer):
            if other.field.modulus != self.field.modulus:
                raise TraceError(
                    f"cannot mix GF({self.field.modulus}) and "
                    f"GF({other.field.modulus}) values in one traced function"
                )
            return other
        if isinstance(other, FieldVal):
            v = other.value
        elif isinstance(other, bool):
            return None
        elif isinstance(other, int):
            v = other % self.field.modulus
        elif isinstance(other, float):
            raise TraceError(
                "floats cannot participate in GF(p) arithmetic; use ints or "
                "field elements"
            )
        else:
            return None
        return self._wrap(Node("const", self.field, const_value=v))

    def _binary(self, op, other, *, reflected=False) -> "Tracer":
        o = self._lift(other)
        if o is None:
            return NotImplemented
        lhs, rhs = (o, self) if reflected else (self, o)
        return self._wrap(Node(op, self.field, args=(lhs.node, rhs.node)))

    def __add__(self, other):
        return self._binary("add", other)

    __radd__ = __add__

    def __mul__(self, other):
        return self._binary("mul", other)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self._binary("sub", other)

    def __rsub__(self, other):
        return self._binary("sub", other, reflected=True)

    def __neg__(self):
        return self._wrap(Node("neg", self.field, args=(self.node,)))

    def inv(self):
        return self._wrap(Node("inv", self.field, args=(self.node,)))

    def __truediv__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return self * o.inv()

    def __bool__(self):
        raise TraceError(
            "a traced value has no concrete truth value: Python control flow "
            "(if/while) cannot depend on kernel inputs. Compute both branches "
            "with field arithmetic, or move the branch outside the @jit "
            "function."
        )

    def __eq__(self, other):
        raise TraceError(
            "comparisons of traced values are not supported inside @jit "
            "functions; compare concrete FieldVal results outside the kernel"
        )

    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__
    __hash__ = object.__hash__

    def __pow__(self, e: int):
        if not isinstance(e, int) or e < 0:
            raise TraceError("only non-negative integer exponents are traceable")
        if e > (1 << 63) - 1:
            raise TraceError("field exponent exceeds the supported 63-bit bound")
        return self._wrap(Node("pow", self.field, args=(self.node,), exponent=e))


def trace(fn, fields) -> Tuple[TraceContext, List[Node], List[Node]]:
    """Trace ``fn`` with symbolic inputs for each field class in ``fields``.

    Returns ``(ctx, input_nodes, output_nodes)``.
    """
    ctx = TraceContext()
    inputs = []
    tracers = []
    for i, f in enumerate(fields):
        n = ctx.add(Node("input", f, index=i))
        inputs.append(n)
        tracers.append(Tracer(n, ctx))

    result = fn(*tracers)

    if isinstance(result, Tracer):
        outputs = [result.node]
    elif isinstance(result, (tuple, list)):
        outputs = []
        for r in result:
            if not isinstance(r, Tracer):
                raise TypeError("traced function must return field tracers")
            outputs.append(r.node)
    else:
        raise TypeError("traced function must return one or more field tracers")

    return ctx, inputs, outputs
