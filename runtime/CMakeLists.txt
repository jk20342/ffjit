cmake_minimum_required(VERSION 3.16)
project(ffjit-runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static runtime library with a stable C ABI. Built with -fPIC so it can be
# linked into shared objects / JIT-loaded code later.
add_library(ff_rt STATIC lib/RuntimeCAPI.cpp)
set_target_properties(ff_rt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ff_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ff_rt PRIVATE -Wall -Wextra)

# Shared build of the same sources, so the Python frontend can dlopen it via
# ctypes (used as an independent inversion oracle in tests).
add_library(ff_rt_shared SHARED lib/RuntimeCAPI.cpp)
set_target_properties(ff_rt_shared PROPERTIES OUTPUT_NAME ff_rt)
target_include_directories(ff_rt_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ff_rt_shared PRIVATE -Wall -Wextra)

enable_testing()

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE ff_rt)
target_compile_options(test_runtime PRIVATE -Wall -Wextra)

add_test(NAME test_runtime COMMAND test_runtime)
