//===- ff-opt.cpp - ffjit optimizer driver --------------------------------===//
//
// Command-line entry point for testing the `field` dialect and its passes,
// mirroring the role of `mlir-opt` / Catalyst's `quantum-opt`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Field/IR/FieldDialect.h"

#ifdef FFJIT_ENABLE_TRANSFORMS
#include "Field/Transforms/Passes.h"
#endif

int main(int argc, char **argv) {
  mlir::registerAllPasses();
#ifdef FFJIT_ENABLE_TRANSFORMS
  ffjit::field::registerFieldPasses();
#endif

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<ffjit::field::FieldDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ffjit optimizer driver\n", registry));
}
