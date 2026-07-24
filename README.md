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

# elliptic curves: jitted Jacobian kernels, GLV endomorphisms, Pippenger MSM
curve, G, r = ff.bn254_g1()
points  = [k * G for k in range(2, 514)]         # k*G uses GLV automatically
scalars = [pow(k, 99, r) for k in range(2, 514)]
S = ff.msm(points, scalars)              # 25x faster than pure Python
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

See [doc/THEORY.md](doc/THEORY.md) for the mathematics (Montgomery's REDC,
Hensel lifting for \(p^{-1} \bmod 2^W\), Fermat inversion) and an annotated
bibliography.

## Architecture

Modeled on [PennyLane Catalyst](https://github.com/PennyLaneAI/catalyst)
(a clone lives in `catalyst/` as a read-only reference):

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
```

## Status

- [x] Phase 0 -- end-to-end pipeline for word-sized primes
- [x] Phase 1 -- multi-limb Montgomery arithmetic for 254-bit+ primes
- [x] Phase 2a -- batched kernels: `FieldArray` + `f.map()` compiled loops
      (7 ns/elem mul on Mersenne61 -- 19x faster than galois; see
      [benchmark/RESULTS.md](benchmark/RESULTS.md))
- [x] Phase 2b -- multi-output kernels, radix-2 NTT with jitted butterflies,
      `Poly` with O(n log n) multiplication (157x over schoolbook at n=4096)
- [x] Phase 3 -- elliptic curves (BN254 G1, secp256k1) with jitted Jacobian
      double/add kernels and Pippenger MSM with batched bucket reduction
- [x] Phase 4a -- GLV endomorphism decomposition (lattice-reduced scalars,
      Straus-Shamir): 1.9x on scalar mult, MSM at 25.2x over pure Python
- [ ] Phase 4b -- batch-affine MSM, pairings, wheels

**Not constant-time.** Like `galois`, this is a research/prototyping tool;
do not use it to handle secret key material.
