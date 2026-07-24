#ifndef FFJIT_FIELD_IR_FIELDOPS_H
#define FFJIT_FIELD_IR_FIELDOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Field/IR/FieldDialect.h"
#include "Field/IR/FieldTypes.h"

#define GET_OP_CLASSES
#include "Field/IR/FieldOps.h.inc"

#endif // FFJIT_FIELD_IR_FIELDOPS_H
