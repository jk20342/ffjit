get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRSupport
  FFJITField
  )

if(FFJIT_ENABLE_TRANSFORMS)
  list(APPEND LIBS FFJITFieldTransforms)
endif()

add_llvm_executable(ff-opt ff-opt.cpp)
llvm_update_compile_flags(ff-opt)
target_link_libraries(ff-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(ff-opt)

if(FFJIT_ENABLE_TRANSFORMS)
  target_compile_definitions(ff-opt PRIVATE FFJIT_ENABLE_TRANSFORMS)
endif()
