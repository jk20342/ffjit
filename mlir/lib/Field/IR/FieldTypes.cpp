#include "Field/IR/FieldTypes.h"
#include "Field/IR/FieldDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ffjit::field;

/// Convenience builder: wrap a raw APInt modulus in an IntegerAttr sized to
/// its own bit width, so that `ElementType::get(ctx, apint)` is ergonomic.
ElementType ElementType::get(::mlir::MLIRContext *ctx, ::llvm::APInt modulus) {
  unsigned width = std::max(1u, modulus.getActiveBits());
  ::llvm::APInt normalized = modulus.zextOrTrunc(width);
  auto attr = IntegerAttr::get(IntegerType::get(ctx, width), normalized);
  return ElementType::get(ctx, attr);
}
