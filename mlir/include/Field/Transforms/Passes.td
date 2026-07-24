#ifndef FFJIT_FIELD_TRANSFORMS_PASSES
#define FFJIT_FIELD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFieldToArith : Pass<"convert-field-to-arith", "::mlir::ModuleOp"> {
  let summary = "Lower the `field` dialect to `arith`/`func` on wide integers.";
  let description = [{
    Converts `!field.elem<p>` to a signless integer of width
    `W = 64 * ceil((bits(p)+1)/64)` and rewrites the field ops to integer
    arithmetic.

    With `montgomery=true` (default) the function body operates in the
    Montgomery domain: values are represented as `a*R mod p` where
    `R = 2^W`. Multiplication uses Montgomery reduction (REDC), which
    replaces the expensive division by `p` with multiplications and a shift
    by `W`. Domain conversions are inserted only at `field.from_int` /
    `field.to_int` boundaries. Inversion uses Fermat's little theorem
    (`a^(p-2)`) with a compile-time-unrolled Montgomery square-and-multiply
    ladder, so the lowering is self-contained.

    With `montgomery=false` the pass emits a straightforward `urem`-based
    reduction, which is slower but serves as a correctness oracle in tests.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"useMontgomery", "montgomery", "bool", /*default=*/"true",
           "Use Montgomery-domain representation and REDC for multiplication.">,
  ];
}

#endif // FFJIT_FIELD_TRANSFORMS_PASSES
