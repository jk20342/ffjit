# ffjit -- a JIT compiler for finite-field arithmetic in Python

`ffjit` is an MLIR- and C++-based just-in-time compiler for Python code over
prime fields \(\mathbb{F}_p\) with **arbitrary-size moduli** -- including the
254-bit and 255-bit primes used by zero-knowledge proof systems (BN254,
BLS12-381) that no existing Python JIT can compile.

```python
import ffjit as ff

F = ff.GF(21888242871839275222246405745257275088548364400416034343698204186575808495617)  # BN254 scalar field

@ff.jit
def horner(x: F, a: F, b: F, c: F):
    # evaluate a*x^2 + b*x + c
    return (a * x + b) * x + c

# scalar call
horner(F(2), F(1), F(3), F(5))

# batched: data stays in native limb buffers, one compiled loop over N
xs = ff.FieldArray(F, range(100_000))
ys = horner.map(xs, xs, xs, xs)          # FieldArray of results

# NTT-based polynomial multiplication (O(n log n), jitted butterflies)
a = ff.Poly(F, range(1, 4097))
b = ff.Poly(F, range(2, 4098))
c = a * b                                # 157x faster than schoolbook

# elliptic curves: jitted kernels, GLV endomorphisms, batch-affine Pippenger MSM
curve, G, r = ff.bn254_g1()              # also: bls12_381_g1(), secp256k1()
points  = [k * G for k in range(2, 514)]         # k*G uses GLV automatically
scalars = [pow(k, 99, r) for k in range(2, 514)]
S = ff.msm(points, scalars)              # 36x faster than pure Python

# fixed-base comb: repeated k*G with zero doublings per multiply
T = G.precompute()                       # 8160-point table, ~40 ms
Q = 123456789 * T                        # 5.8x faster than double-and-add

# negacyclic convolution in GF(p)[x]/(x^n + 1)  (the Ring-LWE ring)
u = ff.FieldArray(F, range(1, 1025))
v = ff.FieldArray(F, range(2, 1026))
w = ff.negacyclic_mul(u, v)
```

## Why

- [`galois`](https://github.com/mhostetter/galois) JIT-compiles field
  arithmetic with Numba -- but only for fields that fit in `int64`. For
  ZK-scale primes it silently falls back to pure-Python object arrays,
  orders of magnitude slower.
- The ZK research workflow today is: prototype in SageMath/Python, then
  hand-port to Rust (arkworks) or C++. `ffjit` aims to make the Python
  prototype fast enough to delay or skip the port.
- Because the modulus is a compile-time constant, a compiler can specialize
  aggressively: multi-limb representation, Montgomery reduction with
  precomputed constants \(p'\), \(R^2 \bmod p\), and (later) NTT-based
  polynomial multiplication and Pippenger multi-scalar multiplication.

## Architecture

```
Python @ff.jit function
  +-- tracer (operator overloading on proxy values)        frontend/ffjit/
       +-- `field` MLIR dialect  (!field.elem<p>)          mlir/include/Field/
            +-- --convert-field-to-arith                   mlir/lib/Field/Transforms/
               Montgomery form + wide-integer arith
                 +-- arith/func -> LLVM dialect -> LLVM IR   tools/ffc driver
                      +-- native object file -> dlopen'd .so
                           +-- C ABI runtime (libff_rt)    runtime/
```

- `frontend/` -- Python package: `GF(p)`, the tracer, `@ff.jit`, compilation
  cache, ctypes marshalling.
- `mlir/` -- out-of-tree MLIR dialect (`field`), lowering passes, the
  `ff-opt` pass-testing tool and the `ffc` ahead-of-time compiler driver.
  Builds against system MLIR (LLVM 21).
- `runtime/` -- C++ runtime with a stable `extern "C"` ABI (big-integer
  modular inverse via binary extended Euclid; future home of hand-tuned
  MSM/NTT kernels).

## Building

Requires: CMake >= 3.20, Ninja, a C++17 compiler, LLVM/MLIR 21 dev packages
(`llvm-21-dev libmlir-21-dev mlir-21-tools`), Python >= 3.10.

```bash
make            # builds mlir tools + runtime, then installs ffjit (editable)
make test       # runtime C++ tests, MLIR lit tests, frontend pytest
make bench      # BN254 benchmark vs galois / pure Python
make perf       # perf-regression check vs a local baseline (make perf-baseline)
```

`ffjit-doctor` (or `python3 -m ffjit._doctor`) diagnoses a broken setup:
it checks for `ffc`, `clang`, numpy, a writable kernel cache, and finishes
with a real end-to-end kernel compile.

## Status

Implemented and benchmarked (see [benchmark/RESULTS.md](benchmark/RESULTS.md)):

- `field` MLIR dialect with Montgomery lowering, canonicalization patterns,
  and compile-time constants specialized per modulus
- scalar + batched (`FieldArray`) + multi-output kernels
- radix-2 NTT with jitted butterflies, `Poly` with O(n log n) multiplication,
  negacyclic convolution in GF(p)[x]/(x^n + 1) via psi-twisting
- elliptic curves (BN254 G1, BLS12-381 G1, secp256k1): jitted Jacobian
  kernels, GLV endomorphism decomposition, batch-affine Pippenger MSM
  (Montgomery shared inversion, cross-window aggregation batching),
  fixed-base comb tables

Deliberately deferred, with reasons:

- **CIOS word-level Montgomery lowering** -- LLVM already legalizes our wide
  `iN` Montgomery arithmetic into limb operations; an explicit CIOS pass in
  the dialect only pays off if it beats that legalization, which needs
  careful measurement before committing to the (large) implementation.
- **Traced loop primitives** (`ff.fori`) -- would let kernels contain
  `scf.for` regions instead of unrolled straight-line code; the batch path
  already covers the data-parallel cases that matter most.
- **Pairings / extension-field towers** (GF(p^2), GF(p^12)) -- a
  project-sized addition; the dialect's type system was designed so
  extension elements can be added without breaking the existing lowering.
