#ifndef FFJIT_FIELD_DIALECT
#define FFJIT_FIELD_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Field dialect definition.
//===----------------------------------------------------------------------===//

def FieldDialect : Dialect {
  let name = "field";
  let summary = "Arithmetic over prime finite fields GF(p) with arbitrary-size moduli.";
  let description = [{
    The `field` dialect models values and operations in a prime field
    (the integers modulo an odd prime `p`). The modulus is carried on the
    type as a compile-time constant, which lets lowering specialize the
    reduction strategy (word-sized `urem` for small primes, multi-limb
    Montgomery reduction for cryptographic 254-bit+ primes).

    Elements are pure values (SSA), so the dataflow of a field program is
    fully exposed for optimization. See `doc/THEORY.md` for the reduction
    algorithms and references.
  }];

  let cppNamespace = "::ffjit::field";
  let useDefaultTypePrinterParser = 1;
  // Canonicalization patterns materialize folded constants as
  // arith.constant feeding field.from_int.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// Base classes for Field ops and types.
//===----------------------------------------------------------------------===//

class Field_Op<string mnemonic, list<Trait> traits = []>
    : Op<FieldDialect, mnemonic, traits>;

class Field_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FieldDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // FFJIT_FIELD_DIALECT
