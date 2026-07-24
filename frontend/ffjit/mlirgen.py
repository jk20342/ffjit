"""Emit textual MLIR (`field` dialect) from a traced DAG.

Each generated module contains both the scalar kernel and a structured batch
entry point.  The batch entry point deliberately uses raw LLVM pointers at its
public boundary while keeping its iteration as ``scf.for`` until ffc lowers
the module.  This preserves the contiguous FieldArray ABI without introducing
memref descriptors.
"""

from __future__ import annotations

from typing import List

from .field import storage_bits
from .tracer import Node, TraceContext


def _elem_ty(f: type) -> str:
    W = storage_bits(f.modulus)
    return f"!field.elem<{f.modulus} : i{W}>"


def _batch_wrapper(name: str, arg_bits: List[int],
                   ret_bits: List[int]) -> str:
    """Generate the raw-pointer batch ABI as structured MLIR."""
    params = ["%n: i64"]
    params.extend(
        f"%out{k}: !llvm.ptr" for k in range(len(ret_bits))
    )
    params.extend(f"%arg{i}: !llvm.ptr" for i in range(len(arg_bits)))

    lines = [f"func.func @{name}_batch({', '.join(params)}) {{"]
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append("  %n_index = arith.index_cast %n : i64 to index")
    for i, width in enumerate(arg_bits):
        lines.append(f"  %arg_stride{i} = arith.constant {width // 8} : i64")
    for k, width in enumerate(ret_bits):
        lines.append(f"  %out_stride{k} = arith.constant {width // 8} : i64")
    lines.append("  scf.for %i = %c0 to %n_index step %c1 {")
    lines.append("    %i64 = arith.index_cast %i : index to i64")

    call_args = []
    for i, width in enumerate(arg_bits):
        lines.append(
            f"    %arg_offset{i} = arith.muli %i64, %arg_stride{i} : i64"
        )
        lines.append(
            f"    %arg_ptr{i} = llvm.getelementptr %arg{i}[%arg_offset{i}] "
            f": (!llvm.ptr, i64) -> !llvm.ptr, i8"
        )
        lines.append(
            f"    %arg_value{i} = llvm.load %arg_ptr{i} "
            f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
        )
        call_args.append(f"%arg_value{i}")

    arg_types = ", ".join(f"i{width}" for width in arg_bits)
    ret_types = ", ".join(f"i{width}" for width in ret_bits)
    if len(ret_bits) == 1:
        lines.append(
            f"    %result = func.call @{name}({', '.join(call_args)}) "
            f": ({arg_types}) -> i{ret_bits[0]}"
        )
    else:
        lines.append(
            f"    %result:{len(ret_bits)} = "
            f"func.call @{name}({', '.join(call_args)}) "
            f": ({arg_types}) -> ({ret_types})"
        )

    for k, width in enumerate(ret_bits):
        result = "%result" if len(ret_bits) == 1 else f"%result#{k}"
        lines.append(
            f"    %out_offset{k} = arith.muli %i64, %out_stride{k} : i64"
        )
        lines.append(
            f"    %out_ptr{k} = llvm.getelementptr %out{k}[%out_offset{k}] "
            f": (!llvm.ptr, i64) -> !llvm.ptr, i8"
        )
        lines.append(
            f"    llvm.store {result}, %out_ptr{k} "
            f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr"
        )

    lines.append("  }")
    lines.append("  return")
    lines.append("}")
    return "\n".join(lines) + "\n"


class GeneratedModule:
    def __init__(self, text: str, name: str, arg_bits: List[int],
                 ret_bits: List[int], requires_runtime: bool = False):
        self.text = text.rstrip() + "\n\n" + _batch_wrapper(
            name, arg_bits, ret_bits
        )
        self.name = name
        self.arg_bits = arg_bits      # storage width per argument
        self.ret_bits = ret_bits      # storage width per result
        self.requires_runtime = requires_runtime


def generate(
    name: str,
    ctx: TraceContext,
    inputs: List[Node],
    outputs: List[Node],
    *,
    requires_runtime: bool = False,
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
        elif n.op == "pow":
            (a,) = n.args
            lines.append(
                f"  {name_i} = field.pow %n{a.id}, {n.exponent} : {ety}"
            )
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

    return GeneratedModule(
        "\n".join(lines) + "\n",
        name,
        arg_bits,
        ret_bits,
        requires_runtime=requires_runtime,
    )
