get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(ffc ffc.cpp)
llvm_update_compile_flags(ffc)
target_link_libraries(ffc PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  FFJITField
  FFJITFieldTransforms
  )
mlir_check_all_link_libraries(ffc)
