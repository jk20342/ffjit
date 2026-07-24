#include "Field/IR/FieldDialect.h"
#include "Field/IR/FieldOps.h"
#include "Field/IR/FieldTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ffjit::field;

#include "Field/IR/FieldOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Field/IR/FieldOpsTypes.cpp.inc"

void FieldDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Field/IR/FieldOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Field/IR/FieldOps.cpp.inc"
      >();
}
