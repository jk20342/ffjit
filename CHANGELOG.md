# Changelog

## Unreleased

Native orchestration and compiler maturity.

- Batch wrappers are structured MLIR `scf.for` loops over the existing raw
  pointer ABI instead of LLVM text appended by the Python compiler.
- Forward and inverse NTT, cyclic polynomial multiplication, and negacyclic
  multiplication have one-call structured native drivers with reference
  fallbacks.
- Batch inversion is a generated one-call MLIR kernel. Runtime ABI v3 moves
  MSM and fixed-base digit, bucket, reduction, and window scheduling out of
  Python while retaining generated field kernels and exceptional-case checks.
- `field.pow` keeps compile-time integer powers compact and lowers them to a
  bounded loop. Field lowering now runs CSE and includes additional algebraic
  canonicalizations.
- `field.inv` can use the default Fermat loop or the C runtime XGCD path with
  explicit Montgomery-domain boundary conversions.
- Added guarded `generic`, `auto`, and `compact` Montgomery-width modes.
  `auto` remains generic because compact results are workload-dependent.
- Runtime and generated-kernel ABI versions now participate in cache keys and
  runtime-required kernels validate and link the matching `libff_rt`.

## 0.2.0

Performance and algorithmic depth.

- Batch-affine Pippenger MSM: inputs normalized to affine with one shared
  (Montgomery-trick) inversion, bucket reduction through a 3-multiplication
  batch-affine add kernel, and running-sum aggregation batched across all
  windows -- MSM over BN254 G1 improved from 25x to 36x over pure Python
  (63 ms at n=512).
- Fixed-base comb precomputation (`Point.precompute()`): table-backed
  scalar multiplication with zero doublings, 5.8x over GLV double-and-add.
- BLS12-381 G1 preset (`ff.bls12_381_g1()`), exercising the 6-limb
  (381-bit) code path end to end; GLV-enabled.
- Negacyclic convolution `ff.negacyclic_mul` in GF(p)[x]/(x^n + 1) via
  psi-twisting (the Ring-LWE ring).
- Poseidon-style hash demo (`demos/poseidon.py`): a full 65-round
  permutation compiled into a single straight-line kernel.
- `ffjit-doctor` console script: toolchain diagnosis ending in a real
  end-to-end kernel compile.
- Perf-regression harness (`make perf` / `make perf-baseline`) with local
  JSON baselines and a slowdown tolerance gate.

## 0.1.0

Initial release.

- `field` MLIR dialect (`!field.elem<p>` with arbitrary-size moduli) with
  `add`/`sub`/`mul`/`neg`/`inv`/`from_int`/`to_int`, `ff-opt`, and the `ffc`
  compiler driver.
- Lowering to `arith`/`scf` with multi-limb Montgomery multiplication (REDC),
  compile-time Newton-Hensel constants, and loop-based Fermat inversion.
- Python frontend: `GF(p)`, operator-overloading tracer, `@ff.jit` with
  scalar, batched (`FieldArray` + `f.map`), and multi-output kernels;
  disk-cached compilation via `clang`.
- Radix-2 NTT with jitted butterflies and `Poly` with O(n log n)
  multiplication.
- Elliptic curves (BN254 G1, secp256k1) with jitted Jacobian kernels,
  GLV endomorphism decomposition, and Pippenger MSM with batched bucket
  reduction.
- C++ runtime (`libff_rt`) with binary extended-Euclidean modular inversion
  used as an independent test oracle.
