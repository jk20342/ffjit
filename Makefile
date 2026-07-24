PYTHON ?= python3
MLIR_DIR ?= /usr/lib/llvm-21/lib/cmake/mlir
LLVM_DIR ?= /usr/lib/llvm-21/lib/cmake/llvm
BUILD_TYPE ?= Release
LIT ?= /usr/lib/llvm-21/bin/lit

.PHONY: all mlir runtime frontend test test-runtime test-mlir test-frontend bench lint format clean

all: mlir runtime frontend

mlir:
	cmake -G Ninja -S mlir -B mlir/build \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DMLIR_DIR=$(MLIR_DIR) -DLLVM_DIR=$(LLVM_DIR)
	cmake --build mlir/build

runtime:
	cmake -G Ninja -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build runtime/build

frontend:
	@echo "The ffjit Python package is pure-Python; either 'pip install -e .'"
	@echo "or run with PYTHONPATH=frontend (the tests do this automatically)."

test: test-runtime test-mlir test-frontend

test-runtime: runtime
	ctest --test-dir runtime/build --output-on-failure

test-mlir: mlir
	$(LIT) -v mlir/test

test-frontend: mlir runtime
	PYTHONPATH=frontend $(PYTHON) -m pytest frontend/test/pytest -q

bench: mlir
	PYTHONPATH=frontend $(PYTHON) benchmark/bench_bn254.py
	PYTHONPATH=frontend $(PYTHON) benchmark/bench_batch.py
	PYTHONPATH=frontend $(PYTHON) benchmark/bench_ntt.py
	PYTHONPATH=frontend $(PYTHON) benchmark/bench_msm.py

demo: mlir
	PYTHONPATH=frontend $(PYTHON) demos/demo.py

lint:
	$(PYTHON) -m ruff check frontend benchmark demos
	find mlir/lib mlir/include mlir/tools runtime \( -name '*.cpp' -o -name '*.h' \) \
		| xargs clang-format --dry-run --Werror

format:
	$(PYTHON) -m ruff check --fix frontend benchmark demos
	find mlir/lib mlir/include mlir/tools runtime \( -name '*.cpp' -o -name '*.h' \) \
		| xargs clang-format -i

clean:
	rm -rf mlir/build runtime/build
