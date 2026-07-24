#ifndef FFJIT_FIELD_TYPES
#define FFJIT_FIELD_TYPES

include "Field/IR/FieldDialect.td"

//===----------------------------------------------------------------------===//
// ElementType : an element of GF(p).
//===----------------------------------------------------------------------===//

def Field_ElementType : Field_Type<"Element", "elem"> {
  let summary = "An element of the prime field GF(p).";
  let description = [{
    `!field.elem<p>` is a residue class modulo the odd prime `p`. The modulus
    is stored as an arbitrary-precision integer attribute, so a single type
    can describe both a 17-bit prime and the 254-bit BN254 scalar field.

    Canonical (non-Montgomery) representatives are integers in `[0, p)`.

    Example:
    ```mlir
    !field.elem<65537 : i32>
    !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
    ```
  }];

  let parameters = (ins "::mlir::IntegerAttr":$modulus);

  let assemblyFormat = "`<` $modulus `>`";

  let builders = [
    TypeBuilder<(ins "::llvm::APInt":$modulus)>,
  ];

  let extraClassDeclaration = [{
    /// The prime modulus as an arbitrary-precision integer.
    ::llvm::APInt getModulusValue() { return getModulus().getValue(); }

    /// Number of significant bits in the modulus.
    unsigned getModulusBitWidth() { return getModulus().getValue().getActiveBits(); }

    /// Number of 64-bit limbs used by the Montgomery representation. This is
    /// the smallest limb count whose bit width strictly exceeds the modulus,
    /// guaranteeing the Montgomery radix R = 2^(64*nlimbs) satisfies R > p.
    unsigned getNumLimbs() {
      unsigned bits = getModulusValue().getActiveBits();
      return (bits / 64) + 1;
    }

    /// Bit width W = 64 * nlimbs of the Montgomery radix exponent.
    unsigned getStorageBitWidth() { return 64u * getNumLimbs(); }
  }];
}

#endif // FFJIT_FIELD_TYPES
