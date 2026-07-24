#ifndef FFJIT_FIELD_OPS
#define FFJIT_FIELD_OPS

include "Field/IR/FieldDialect.td"
include "Field/IR/FieldTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Binary arithmetic ops.
//===----------------------------------------------------------------------===//

class Field_BinaryOp<string mnemonic, list<Trait> traits = []>
    : Field_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Field_ElementType:$lhs, Field_ElementType:$rhs);
  let results = (outs Field_ElementType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Field_AddOp : Field_BinaryOp<"add", [Commutative]> {
  let summary = "Field addition (a + b) mod p.";
  let hasCanonicalizer = 1;
}

def Field_SubOp : Field_BinaryOp<"sub"> {
  let summary = "Field subtraction (a - b) mod p.";
  let hasCanonicalizer = 1;
}

def Field_MulOp : Field_BinaryOp<"mul", [Commutative]> {
  let summary = "Field multiplication (a * b) mod p.";
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Unary ops.
//===----------------------------------------------------------------------===//

class Field_UnaryOp<string mnemonic, list<Trait> traits = []>
    : Field_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Field_ElementType:$operand);
  let results = (outs Field_ElementType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def Field_NegOp : Field_UnaryOp<"neg"> {
  let summary = "Field negation (-a) mod p.";
  let hasCanonicalizer = 1;
}

def Field_InvOp : Field_UnaryOp<"inv"> {
  let summary = "Multiplicative inverse a^(-1) mod p.";
  let hasCanonicalizer = 1;
  let description = [{
    Computes the multiplicative inverse in GF(p). By convention `inv(0) = 0`.
    The default lowering is a self-contained pure-IR Fermat exponentiation.
    An optional lowering calls the versioned C ABI runtime.
  }];
}

def Field_PowOp : Field_Op<"pow", [Pure, SameOperandsAndResultType]> {
  let summary = "Raise a field element to a bounded constant exponent.";
  let description = [{
    Computes `base^exponent` in GF(p), where `exponent` is a non-negative
    compile-time integer in the signed 64-bit range. Lowering emits a compact
    square-and-multiply loop rather than expanding the exponentiation in the
    traced graph.
  }];
  let arguments = (ins Field_ElementType:$base, I64Attr:$exponent);
  let results = (outs Field_ElementType:$result);
  let assemblyFormat =
      "$base `,` $exponent attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Conversion ops (boundary between machine integers and field elements).
//===----------------------------------------------------------------------===//

def Field_FromIntOp : Field_Op<"from_int", [Pure]> {
  let summary = "Reduce a machine integer into GF(p).";
  let description = [{
    Interprets a signless integer as an unsigned value and reduces it modulo
    `p` to a field element. This is the entry boundary; when the body is
    compiled in the Montgomery domain, this op also multiplies by R^2 and
    applies REDC to move into that domain.
  }];
  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs Field_ElementType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Field_ToIntOp : Field_Op<"to_int", [Pure]> {
  let summary = "Extract the canonical representative in [0, p) as an integer.";
  let description = [{
    Produces the canonical integer representative (in `[0, p)`) of a field
    element, zero-extended/truncated to the requested signless integer type.
    This is the exit boundary; it undoes any Montgomery-domain encoding.
  }];
  let arguments = (ins Field_ElementType:$input);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // FFJIT_FIELD_OPS
