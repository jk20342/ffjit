"""Emit textual MLIR (`field` dialect) from a traced DAG."""

from __future__ import annotations

from typing import List

from .field import storage_bits
from .tracer import Node, TraceContext


def _elem_ty(f: type) -> str:
    W = storage_bits(f.modulus)
    return f"!field.elem<{f.modulus} : i{W}>"


class GeneratedModule:
    def __init__(self, text: str, name: str, arg_bits: List[int],
                 ret_bits: List[int]):
        self.text = text
        self.name = name
        self.arg_bits = arg_bits      # storage width per argument
        self.ret_bits = ret_bits      # storage width per result


def generate(
    name: str,
    ctx: TraceContext,
    inputs: List[Node],
    outputs: List[Node],
) -> GeneratedModule:
    arg_bits = [storage_bits(n.field.modulus) for n in inputs]
    ret_bits = [storage_bits(o.field.modulus) for o in outputs]

    ssa = {}  # node.id -> ssa name
    lines: List[str] = []

    params = ", ".join(f"%arg{n.index}: i{arg_bits[i]}" for i, n in enumerate(inputs))
    if len(ret_bits) == 1:
        ret_ty = f"i{ret_bits[0]}"
    else:
        ret_ty = "(" + ", ".join(f"i{w}" for w in ret_bits) + ")"
    lines.append(f"func.func @{name}({params}) -> {ret_ty} {{")

    for n in ctx.nodes:
        ety = _elem_ty(n.field)
        name_i = f"%n{n.id}"
        if n.op == "input":
            W = storage_bits(n.field.modulus)
            lines.append(
                f"  {name_i} = field.from_int %arg{n.index} : i{W} -> {ety}"
            )
        elif n.op == "const":
            W = storage_bits(n.field.modulus)
            lines.append(f"  %c{n.id} = arith.constant {n.const_value} : i{W}")
            lines.append(f"  {name_i} = field.from_int %c{n.id} : i{W} -> {ety}")
        elif n.op in ("add", "sub", "mul"):
            a, b = n.args
            lines.append(
                f"  {name_i} = field.{n.op} %n{a.id}, %n{b.id} : {ety}"
            )
        elif n.op in ("neg", "inv"):
            (a,) = n.args
            lines.append(f"  {name_i} = field.{n.op} %n{a.id} : {ety}")
        else:
            raise ValueError(f"unknown node op {n.op!r}")
        ssa[n.id] = name_i

    ret_names = []
    for k, out in enumerate(outputs):
        oety = _elem_ty(out.field)
        lines.append(
            f"  %ret{k} = field.to_int %n{out.id} : {oety} -> i{ret_bits[k]}"
        )
        ret_names.append(f"%ret{k}")
    lines.append(
        f"  return {', '.join(ret_names)} : "
        + ", ".join(f"i{w}" for w in ret_bits)
    )
    lines.append("}")

    return GeneratedModule("\n".join(lines) + "\n", name, arg_bits, ret_bits)
