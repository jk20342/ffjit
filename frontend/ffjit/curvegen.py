"""Generate structured MLIR helpers for elliptic-curve batches."""

from __future__ import annotations

from .compiler import RawPointerModule
from .field import storage_bits


def _ptr(lines, name: str, base: str, offset: str, indent: str = "    ") -> None:
    lines.append(
        f"{indent}{name} = llvm.getelementptr {base}[{offset}] "
        f": (!llvm.ptr, i64) -> !llvm.ptr, i8"
    )


def generate_batch_inv_module(name: str, modulus: int, n: int) -> RawPointerModule:
    """Generate zero-preserving Montgomery batch inversion for ``n`` values."""
    if n <= 0:
        raise ValueError("batch inversion length must be positive")
    width = storage_bits(modulus)
    elem = f"!field.elem<{modulus} : i{width}>"
    lines = [
        f"func.func @{name}(%out: !llvm.ptr, %input: !llvm.ptr, "
        "%scratch: !llvm.ptr) {",
        "  %c0 = arith.constant 0 : index",
        "  %c1 = arith.constant 1 : index",
        f"  %cn = arith.constant {n} : index",
        f"  %cn64 = arith.constant {n} : i64",
        f"  %stride = arith.constant {width // 8} : i64",
        f"  %zero_int = arith.constant 0 : i{width}",
        f"  %one_int = arith.constant 1 : i{width}",
        f"  %product_int = scf.for %i = %c0 to %cn step %c1 "
        f"iter_args(%prefix_int = %one_int) -> (i{width}) {{",
        "    %i64 = arith.index_cast %i : index to i64",
        "    %offset = arith.muli %i64, %stride : i64",
    ]
    _ptr(lines, "%input_ptr", "%input", "%offset")
    _ptr(lines, "%scratch_ptr", "%scratch", "%offset")
    lines.extend(
        [
            f"    %input_int = llvm.load %input_ptr "
            f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
            f"    %is_zero = arith.cmpi eq, %input_int, %zero_int : i{width}",
            f"    %factor_int = arith.select %is_zero, %one_int, %input_int "
            f": i{width}",
            f"    %factor = field.from_int %factor_int : i{width} -> {elem}",
            f"    %prefix = field.from_int %prefix_int : i{width} -> {elem}",
            f"    llvm.store %prefix_int, %scratch_ptr "
            f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr",
            f"    %next = field.mul %prefix, %factor : {elem}",
            f"    %next_int = field.to_int %next : {elem} -> i{width}",
            f"    scf.yield %next_int : i{width}",
            "  }",
            f"  %product = field.from_int %product_int : i{width} -> {elem}",
            f"  %inverse = field.inv %product : {elem}",
            f"  %inverse_int = field.to_int %inverse : {elem} -> i{width}",
            f"  %final_int = scf.for %j = %c0 to %cn step %c1 "
            f"iter_args(%suffix_int = %inverse_int) -> (i{width}) {{",
            "    %j64 = arith.index_cast %j : index to i64",
            "    %rev0 = arith.subi %cn64, %j64 : i64",
            "    %one64 = arith.constant 1 : i64",
            "    %rev = arith.subi %rev0, %one64 : i64",
            "    %offset2 = arith.muli %rev, %stride : i64",
        ]
    )
    _ptr(lines, "%input_ptr2", "%input", "%offset2")
    _ptr(lines, "%scratch_ptr2", "%scratch", "%offset2")
    _ptr(lines, "%out_ptr", "%out", "%offset2")
    lines.extend(
        [
            f"    %input_int2 = llvm.load %input_ptr2 "
            f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
            f"    %prefix_int2 = llvm.load %scratch_ptr2 "
            f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
            f"    %prefix2 = field.from_int %prefix_int2 : i{width} -> {elem}",
            f"    %suffix = field.from_int %suffix_int : i{width} -> {elem}",
            f"    %is_zero2 = arith.cmpi eq, %input_int2, %zero_int : i{width}",
            f"    %candidate = field.mul %prefix2, %suffix : {elem}",
            f"    %candidate_int = field.to_int %candidate : {elem} -> i{width}",
            f"    %result_int = arith.select %is_zero2, %zero_int, "
            f"%candidate_int : i{width}",
            f"    llvm.store %result_int, %out_ptr "
            f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr",
            f"    %factor_int2 = arith.select %is_zero2, %one_int, "
            f"%input_int2 : i{width}",
            f"    %factor2 = field.from_int %factor_int2 : i{width} -> {elem}",
            f"    %next_suffix = field.mul %suffix, %factor2 : {elem}",
            f"    %next_suffix_int = field.to_int %next_suffix : {elem} -> i{width}",
            f"    scf.yield %next_suffix_int : i{width}",
            "  }",
            "  return",
            "}",
        ]
    )
    return RawPointerModule("\n".join(lines) + "\n", name, 3)
