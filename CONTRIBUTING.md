# Contributing

## Prerequisites

- CMake >= 3.20, Ninja, a C++17 compiler
- LLVM/MLIR 21 development packages:
  `sudo apt install llvm-21-dev libmlir-21-dev mlir-21-tools clang`
- Python >= 3.10 with `pytest`, `hypothesis` (and optionally `numpy`)

## Building and testing

```bash
make            # build MLIR tools (ff-opt, ffc) and the C++ runtime
make test       # ctest (runtime) + lit (dialect/lowering) + pytest (frontend)
make demo       # end-to-end tour
make bench      # benchmark suite (see benchmark/RESULTS.md)
make lint       # ruff + clang-format checks
```

The Python package is pure Python: either `pip install -e .` or run with
`PYTHONPATH=frontend` (the Makefile test targets do the latter).

## Layout

- `frontend/ffjit/` -- tracer, MLIR generation, compilation cache, ctypes
  marshalling, NTT/Poly/curve layers
- `mlir/` -- the `field` dialect, `convert-field-to-arith` (Montgomery
  lowering), `ff-opt`, `ffc`
- `runtime/` -- C ABI runtime library (independent oracle for tests)
- `frontend/test/pytest/`, `mlir/test/`, `runtime/tests/` -- the three suites

## Conventions

- Every arithmetic feature needs a differential test against pure-Python
  big-int reference behavior (`hypothesis` where practical).
- Changes to the lowering need lit/FileCheck coverage in `mlir/test/Field/`.
- Kernel ABI changes must bump the `abi=N` tag in
  `frontend/ffjit/compiler.py` so stale cached objects are invalidated.
- Python is linted with `ruff`, C++ with `clang-format` (LLVM style).
