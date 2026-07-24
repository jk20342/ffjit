//===- ffc.cpp - ffjit ahead-of-time compiler driver ----------------------===//
//
// Reads a `.mlir` module using the `field` dialect, lowers it through
// `convert-field-to-arith` and the standard arith/func -> LLVM pipeline, and
// emits LLVM IR text (which the Python frontend compiles to a shared object
// with clang) or the post-lowering MLIR for debugging.
//
// Usage:
//   ffc input.mlir -o out.ll [--emit=llvm|mlir] [--no-montgomery]
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "Field/IR/FieldDialect.h"
#include "Field/Transforms/Passes.h"

#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  std::string inputFile, outputFile = "-", emit = "llvm";
  bool montgomery = true;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-o" && i + 1 < argc) {
      outputFile = argv[++i];
    } else if (a.rfind("--emit=", 0) == 0) {
      emit = a.substr(7);
    } else if (a == "--no-montgomery") {
      montgomery = false;
    } else if (a == "-h" || a == "--help") {
      llvm::errs() << "usage: ffc input.mlir -o out.ll "
                      "[--emit=llvm|mlir] [--no-montgomery]\n";
      return 0;
    } else if (a[0] != '-') {
      inputFile = a;
    } else {
      llvm::errs() << "ffc: unknown flag '" << a << "'\n";
      return 1;
    }
  }
  if (inputFile.empty()) {
    llvm::errs() << "ffc: no input file\n";
    return 1;
  }

  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registry.insert<ffjit::field::FieldDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  registerAllPasses();
  ffjit::field::registerFieldPasses();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFile, &context);
  if (!module) {
    llvm::errs() << "ffc: failed to parse " << inputFile << "\n";
    return 1;
  }

  std::string montStr = montgomery ? "true" : "false";
  // canonicalize first: field-level algebraic identities (x*1, x+0, x*0,
  // constant folding) are far cheaper to apply before each op expands into
  // wide-integer Montgomery arithmetic.
  std::string pipeline = "canonicalize,convert-field-to-arith{montgomery=" +
                         montStr +
                         "},convert-scf-to-cf,convert-arith-to-llvm,"
                         "convert-cf-to-llvm,convert-func-to-llvm,"
                         "reconcile-unrealized-casts";

  PassManager pm(&context);
  if (failed(parsePassPipeline(pipeline, pm))) {
    llvm::errs() << "ffc: failed to build pass pipeline\n";
    return 1;
  }
  if (failed(pm.run(*module))) {
    llvm::errs() << "ffc: lowering failed\n";
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputFile, ec,
                           outputFile == "-" ? llvm::sys::fs::OF_Text
                                             : llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "ffc: cannot open output '" << outputFile
                 << "': " << ec.message() << "\n";
    return 1;
  }

  if (emit == "mlir") {
    module->print(out);
    return 0;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*module, llvmContext, "ffjit");
  if (!llvmModule) {
    llvm::errs() << "ffc: translation to LLVM IR failed\n";
    return 1;
  }
  llvmModule->print(out, nullptr);
  return 0;
}
