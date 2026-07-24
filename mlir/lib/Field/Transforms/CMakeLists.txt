add_mlir_dialect_library(FFJITFieldTransforms
  ConvertFieldToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Field

  DEPENDS
  FFJITFieldTransformsIncGen
  FFJITFieldOpsIncGen

  LINK_LIBS PUBLIC
  FFJITField
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRSCFDialect
  )
