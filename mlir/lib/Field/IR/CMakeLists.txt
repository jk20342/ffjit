add_mlir_dialect_library(FFJITField
  FieldDialect.cpp
  FieldTypes.cpp
  FieldOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Field

  DEPENDS
  FFJITFieldOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRArithDialect
  )
