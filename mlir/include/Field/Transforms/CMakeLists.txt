set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Field)
add_public_tablegen_target(FFJITFieldTransformsIncGen)

add_mlir_doc(Passes FieldPasses ./ -gen-pass-doc)
