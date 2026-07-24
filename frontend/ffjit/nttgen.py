"""Generate fixed-size structured MLIR for native NTT operations."""

from __future__ import annotations

from .compiler import RawPointerModule
from .field import storage_bits


def _elem_ty(p: int, width: int) -> str:
    return f"!field.elem<{p} : i{width}>"


def _function_constants(lines, n: int, width: int) -> None:
    lines.extend(
        [
            "  %c0 = arith.constant 0 : index",
            "  %c1 = arith.constant 1 : index",
            f"  %cn = arith.constant {n} : index",
            f"  %stride = arith.constant {width // 8} : i64",
        ]
    )


def _emit_ptr(lines, result: str, base: str, offset: str, indent: str) -> None:
    lines.append(
        f"{indent}{result} = llvm.getelementptr {base}[{offset}] "
        f": (!llvm.ptr, i64) -> !llvm.ptr, i8"
    )


def _emit_copy(lines, dst: str, src: str, width: int, tag: str) -> None:
    lines.append(f"  scf.for %{tag}_i = %c0 to %cn step %c1 {{")
    lines.append(f"    %{tag}_i64 = arith.index_cast %{tag}_i : index to i64")
    lines.append(f"    %{tag}_off = arith.muli %{tag}_i64, %stride : i64")
    _emit_ptr(lines, f"%{tag}_src", src, f"%{tag}_off", "    ")
    _emit_ptr(lines, f"%{tag}_dst", dst, f"%{tag}_off", "    ")
    lines.append(
        f"    %{tag}_value = llvm.load %{tag}_src "
        f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
    )
    lines.append(
        f"    llvm.store %{tag}_value, %{tag}_dst "
        f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr"
    )
    lines.append("  }")


def _emit_field_mul_loop(
    lines,
    dst: str,
    lhs: str,
    rhs: str,
    p: int,
    width: int,
    tag: str,
) -> None:
    elem = _elem_ty(p, width)
    lines.append(f"  scf.for %{tag}_i = %c0 to %cn step %c1 {{")
    lines.append(f"    %{tag}_i64 = arith.index_cast %{tag}_i : index to i64")
    lines.append(f"    %{tag}_off = arith.muli %{tag}_i64, %stride : i64")
    _emit_ptr(lines, f"%{tag}_lhs_ptr", lhs, f"%{tag}_off", "    ")
    _emit_ptr(lines, f"%{tag}_rhs_ptr", rhs, f"%{tag}_off", "    ")
    _emit_ptr(lines, f"%{tag}_dst_ptr", dst, f"%{tag}_off", "    ")
    lines.append(
        f"    %{tag}_lhs_int = llvm.load %{tag}_lhs_ptr "
        f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
    )
    lines.append(
        f"    %{tag}_rhs_int = llvm.load %{tag}_rhs_ptr "
        f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
    )
    lines.append(f"    %{tag}_lhs = field.from_int %{tag}_lhs_int : i{width} -> {elem}")
    lines.append(f"    %{tag}_rhs = field.from_int %{tag}_rhs_int : i{width} -> {elem}")
    lines.append(f"    %{tag}_product = field.mul %{tag}_lhs, %{tag}_rhs : {elem}")
    lines.append(f"    %{tag}_out = field.to_int %{tag}_product : {elem} -> i{width}")
    lines.append(
        f"    llvm.store %{tag}_out, %{tag}_dst_ptr "
        f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr"
    )
    lines.append("  }")


def _emit_bit_reversal(lines, buf: str, logn: int, width: int) -> None:
    lines.append("  scf.for %br_i = %c0 to %cn step %c1 {")
    lines.append("    %br_i64 = arith.index_cast %br_i : index to i64")
    lines.append("    %br_rev0 = arith.constant 0 : i64")
    previous = "%br_rev0"
    for bit in range(logn):
        lines.append(f"    %br_in_shift{bit} = arith.constant {bit} : i64")
        lines.append(
            f"    %br_shifted{bit} = arith.shrui %br_i64, %br_in_shift{bit} : i64"
        )
        lines.append(f"    %br_one{bit} = arith.constant 1 : i64")
        lines.append(
            f"    %br_bit{bit} = arith.andi %br_shifted{bit}, %br_one{bit} : i64"
        )
        lines.append(f"    %br_out_shift{bit} = arith.constant {logn - 1 - bit} : i64")
        lines.append(
            f"    %br_placed{bit} = arith.shli %br_bit{bit}, %br_out_shift{bit} : i64"
        )
        lines.append(
            f"    %br_rev{bit + 1} = arith.ori {previous}, %br_placed{bit} : i64"
        )
        previous = f"%br_rev{bit + 1}"
    lines.append(f"    %br_swap = arith.cmpi ult, %br_i64, {previous} : i64")
    lines.append("    scf.if %br_swap {")
    lines.append("      %br_left_off = arith.muli %br_i64, %stride : i64")
    lines.append(f"      %br_right_off = arith.muli {previous}, %stride : i64")
    _emit_ptr(lines, "%br_left_ptr", buf, "%br_left_off", "      ")
    _emit_ptr(lines, "%br_right_ptr", buf, "%br_right_off", "      ")
    lines.append(
        f"      %br_left = llvm.load %br_left_ptr "
        f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
    )
    lines.append(
        f"      %br_right = llvm.load %br_right_ptr "
        f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}"
    )
    lines.append(
        f"      llvm.store %br_right, %br_left_ptr "
        f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr"
    )
    lines.append(
        f"      llvm.store %br_left, %br_right_ptr "
        f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr"
    )
    lines.append("    }")
    lines.append("  }")


def _transform_function(
    name: str,
    p: int,
    logn: int,
    *,
    inverse: bool,
) -> str:
    n = 1 << logn
    width = storage_bits(p)
    elem = _elem_ty(p, width)
    lines = [f"func.func private @{name}(%buf: !llvm.ptr, %tw: !llvm.ptr) {{"]
    _function_constants(lines, n, width)
    if inverse:
        ninv = pow(n, -1, p)
        lines.append(f"  %ninv_int = arith.constant {ninv} : i{width}")
        lines.append(f"  %ninv = field.from_int %ninv_int : i{width} -> {elem}")
    _emit_bit_reversal(lines, "%buf", logn, width)

    twiddle_base = 0
    for stage in range(1, logn + 1):
        block = 1 << stage
        half = block >> 1
        tag = f"s{stage}"
        lines.extend(
            [
                f"  %{tag}_end = arith.constant {n // 2} : index",
                f"  %{tag}_half = arith.constant {half} : i64",
                f"  %{tag}_block = arith.constant {block} : i64",
                f"  %{tag}_tw_base = arith.constant {twiddle_base} : i64",
                f"  scf.for %{tag}_i = %c0 to %{tag}_end step %c1 {{",
                f"    %{tag}_i64 = arith.index_cast %{tag}_i : index to i64",
                f"    %{tag}_group = arith.divui %{tag}_i64, %{tag}_half : i64",
                f"    %{tag}_j = arith.remui %{tag}_i64, %{tag}_half : i64",
                f"    %{tag}_group_start = arith.muli %{tag}_group, %{tag}_block : i64",
                f"    %{tag}_left = arith.addi %{tag}_group_start, %{tag}_j : i64",
                f"    %{tag}_right = arith.addi %{tag}_left, %{tag}_half : i64",
                f"    %{tag}_tw_index = arith.addi %{tag}_tw_base, %{tag}_j : i64",
                f"    %{tag}_left_off = arith.muli %{tag}_left, %stride : i64",
                f"    %{tag}_right_off = arith.muli %{tag}_right, %stride : i64",
                f"    %{tag}_tw_off = arith.muli %{tag}_tw_index, %stride : i64",
            ]
        )
        _emit_ptr(lines, f"%{tag}_left_ptr", "%buf", f"%{tag}_left_off", "    ")
        _emit_ptr(lines, f"%{tag}_right_ptr", "%buf", f"%{tag}_right_off", "    ")
        _emit_ptr(lines, f"%{tag}_tw_ptr", "%tw", f"%{tag}_tw_off", "    ")
        lines.extend(
            [
                f"    %{tag}_a_int = llvm.load %{tag}_left_ptr "
                f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
                f"    %{tag}_b_int = llvm.load %{tag}_right_ptr "
                f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
                f"    %{tag}_w_int = llvm.load %{tag}_tw_ptr "
                f"{{alignment = 8 : i64}} : !llvm.ptr -> i{width}",
                f"    %{tag}_a = field.from_int %{tag}_a_int : i{width} -> {elem}",
                f"    %{tag}_b = field.from_int %{tag}_b_int : i{width} -> {elem}",
                f"    %{tag}_w = field.from_int %{tag}_w_int : i{width} -> {elem}",
                f"    %{tag}_t = field.mul %{tag}_w, %{tag}_b : {elem}",
                f"    %{tag}_u0 = field.add %{tag}_a, %{tag}_t : {elem}",
                f"    %{tag}_v0 = field.sub %{tag}_a, %{tag}_t : {elem}",
            ]
        )
        u, v = f"%{tag}_u0", f"%{tag}_v0"
        if inverse and stage == logn:
            lines.extend(
                [
                    f"    %{tag}_u = field.mul %{tag}_u0, %ninv : {elem}",
                    f"    %{tag}_v = field.mul %{tag}_v0, %ninv : {elem}",
                ]
            )
            u, v = f"%{tag}_u", f"%{tag}_v"
        lines.extend(
            [
                f"    %{tag}_u_int = field.to_int {u} : {elem} -> i{width}",
                f"    %{tag}_v_int = field.to_int {v} : {elem} -> i{width}",
                f"    llvm.store %{tag}_u_int, %{tag}_left_ptr "
                f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr",
                f"    llvm.store %{tag}_v_int, %{tag}_right_ptr "
                f"{{alignment = 8 : i64}} : i{width}, !llvm.ptr",
                "  }",
            ]
        )
        twiddle_base += half

    lines.extend(["  return", "}"])
    return "\n".join(lines)


def generate_ntt_module(
    name: str,
    p: int,
    logn: int,
    *,
    inverse: bool,
) -> RawPointerModule:
    n = 1 << logn
    width = storage_bits(p)
    helper = f"{name}_transform"
    lines = [
        _transform_function(helper, p, logn, inverse=inverse),
        "",
        f"func.func @{name}(%out: !llvm.ptr, %input: !llvm.ptr, %tw: !llvm.ptr) {{",
    ]
    _function_constants(lines, n, width)
    _emit_copy(lines, "%out", "%input", width, "copy")
    lines.extend(
        [
            f"  func.call @{helper}(%out, %tw) : (!llvm.ptr, !llvm.ptr) -> ()",
            "  return",
            "}",
        ]
    )
    return RawPointerModule("\n".join(lines) + "\n", name, 3)


def generate_mul_module(
    name: str,
    p: int,
    logn: int,
    *,
    negacyclic: bool,
) -> RawPointerModule:
    n = 1 << logn
    width = storage_bits(p)
    forward = f"{name}_forward"
    inverse = f"{name}_inverse"
    params = [
        "%out: !llvm.ptr",
        "%a: !llvm.ptr",
        "%b: !llvm.ptr",
        "%scratch_a: !llvm.ptr",
        "%scratch_b: !llvm.ptr",
        "%tw_fwd: !llvm.ptr",
        "%tw_inv: !llvm.ptr",
    ]
    if negacyclic:
        params.extend(["%psi: !llvm.ptr", "%psi_inv: !llvm.ptr"])
    lines = [
        _transform_function(forward, p, logn, inverse=False),
        "",
        _transform_function(inverse, p, logn, inverse=True),
        "",
        f"func.func @{name}({', '.join(params)}) {{",
    ]
    _function_constants(lines, n, width)
    if negacyclic:
        _emit_field_mul_loop(lines, "%scratch_a", "%a", "%psi", p, width, "twist_a")
        _emit_field_mul_loop(lines, "%scratch_b", "%b", "%psi", p, width, "twist_b")
    else:
        _emit_copy(lines, "%scratch_a", "%a", width, "copy_a")
        _emit_copy(lines, "%scratch_b", "%b", width, "copy_b")
    lines.extend(
        [
            f"  func.call @{forward}(%scratch_a, %tw_fwd) "
            ": (!llvm.ptr, !llvm.ptr) -> ()",
            f"  func.call @{forward}(%scratch_b, %tw_fwd) "
            ": (!llvm.ptr, !llvm.ptr) -> ()",
        ]
    )
    _emit_field_mul_loop(
        lines, "%out", "%scratch_a", "%scratch_b", p, width, "pointwise"
    )
    lines.append(
        f"  func.call @{inverse}(%out, %tw_inv) : (!llvm.ptr, !llvm.ptr) -> ()"
    )
    if negacyclic:
        _emit_field_mul_loop(lines, "%out", "%out", "%psi_inv", p, width, "untwist")
    lines.extend(["  return", "}"])
    nargs = 9 if negacyclic else 7
    return RawPointerModule("\n".join(lines) + "\n", name, nargs)
