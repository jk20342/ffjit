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
    by `W`. Domain conversions are inserted at integer boundaries and around
    runtime calls. Inversion defaults to Fermat's little theorem (`a^(p-2)`)
    with a pure-IR square-and-multiply `scf.for` loop. With `inv=runtime`,
    inversion converts its operand to canonical form, calls `ff_rt_inv`, and
    converts the result back to Montgomery form.

    With `montgomery=false` the pass emits a straightforward `urem`-based
    reduction, which is slower but serves as a correctness oracle in tests.

    `limb-specialization=compact` opts supported storage widths into a
    `2W+1` Montgomery intermediate after validating the modulus range and
    parity. The generic path uses `2W+64`. `auto` currently resolves to
    generic because benchmarks did not establish a consistent compact win.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"useMontgomery", "montgomery", "bool", /*default=*/"true",
           "Use Montgomery-domain representation and REDC for multiplication.">,
    Option<"invMethod", "inv", "std::string", /*default=*/"\"fermat\"",
           "Inversion implementation: fermat or runtime.">,
    Option<"limbSpecialization", "limb-specialization", "std::string",
           /*default=*/"\"auto\"",
           "Wide-integer lowering: generic, auto, or compact.">,
  ];
}

#endif // FFJIT_FIELD_TRANSFORMS_PASSES
