cmake_minimum_required(VERSION 3.20)
project(ffjit-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The lowering passes and the ffc driver are the core of the project; the
# switch exists only for bisecting dialect-only build issues.
option(FFJIT_ENABLE_TRANSFORMS
       "Build the field-to-arith transforms and the ffc driver" ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/Field/IR)
add_subdirectory(lib/Field/IR)
add_subdirectory(tools/ff-opt)
if(FFJIT_ENABLE_TRANSFORMS)
  add_subdirectory(include/Field/Transforms)
  add_subdirectory(lib/Field/Transforms)
  add_subdirectory(tools/ffc)
endif()
