# Benchmark results

## Elliptic-curve multi-scalar multiplication (Phase 3)

Workload: \(\sum_i k_i P_i\) over **BN254 G1** with random points and full
254-bit scalars. ffjit uses Pippenger's bucket method, *fully batched*: all
inputs are normalized to affine with one shared inversion, bucket reduction
runs through a 3-multiplication batch-affine add kernel (Montgomery shared
inversion for the slopes), and the running-sum aggregation is batched
across all windows simultaneously. References: the same compiled kernels
with the naive per-point double-and-add algorithm, and pure-Python affine
double-and-add (the typical prototype code). Run with
`PYTHONPATH=frontend python3 benchmark/bench_msm.py`.

| N | ffjit Pippenger+GLV | per-point (jitted, GLV) | pure Python | speedup |
|---|---|---|---|---|
| 128 | 22 ms | 322 ms | 556 ms | 25.4x |
| 512 | 63 ms | 1.32 s | 2.28 s | **36.0x** |
| 2048 | 209 ms (102 us/pt) | -- | -- | -- |

Points of interest:

- **Batch-affine arithmetic**: Montgomery's shared-inversion trick turns n
  modular inversions into 3(n-1) multiplications + 1 inversion, so bucket
  adds run in affine coordinates at 3 kernel multiplications each instead
  of ~16 for Jacobian.
- **Cross-window aggregation batching**: the classical per-window
  running-sum aggregation is ~2*2^c *sequential* kernel calls per window.
  Processing all windows' buckets simultaneously turns these into ~2*2^c
  *batch* calls total (each covering every window at once), which roughly
  doubled MSM throughput and moved the optimal window back up to
  \(c \approx \log_2 n - 3\) (measured: c=5 at 256 pairs, c=9 at 4096).
- **GLV endomorphism decomposition** (automatic for BN254 G1, BLS12-381 G1
  and secp256k1, all j-invariant 0): each scalar splits as
  \(k = k_1 + k_2\lambda\) with \(|k_i| \lesssim \sqrt{r}\) via a lattice
  basis found by the extended Euclidean algorithm. Scalar multiplication
  gains ~1.9x (Straus-Shamir joint evaluation halves the doublings). See
  `doc/THEORY.md`.

### Fixed-base scalar multiplication (comb precomputation)

For a repeatedly used base point (`G.precompute(window=8)`), a Lim-Lee comb
table T[j][d] = d * 2^(8j) * G (8160 points, built in ~40 ms with batched
adds and stored affine via one shared inversion) reduces every subsequent
multiplication to <= 32 additions with **zero doublings**:

| method | per k*G |
|---|---|
| comb table (c=8) | **0.43 ms** |
| GLV + Straus-Shamir double-and-add | 2.53 ms (5.8x slower) |

## NTT polynomial multiplication (Phase 2b)

Workload: multiply two random degree-(n-1) polynomials over the **BN254
scalar field**. ffjit uses a radix-2 Cooley-Tukey NTT whose butterflies
\((a, b, \omega) \mapsto (a + \omega b, a - \omega b)\) run in a JIT-compiled
multi-output batch kernel (Montgomery arithmetic, native limb buffers); the
reference is O(n^2) schoolbook in CPython big ints. Run with
`PYTHONPATH=frontend python3 benchmark/bench_ntt.py`.

| n | ffjit NTT | schoolbook (Python) | speedup |
|---|---|---|---|
| 512 | 3.3 ms | 76 ms | 23x |
| 1024 | 7.0 ms | 306 ms | 44x |
| 2048 | 14.8 ms | 1.24 s | 84x |
| 4096 | 31.2 ms | 4.91 s | **157x** |
| 8192 | 65.8 ms | -- | (reference too slow) |

The ffjit column doubles as n doubles -- the expected O(n log n). `galois`
cannot participate at all: constructing GF(p) for BN254 requires factoring
p-1 (ffjit finds roots of unity without factoring; see doc/THEORY.md).

## Batched kernels (Phase 2a): `FieldArray` + `f.map(...)`

Workload: elementwise kernels over N = 100 000 pre-marshalled elements. Data
lives in native limb buffers (`FieldArray`); the compiled batch loop runs
directly over the buffers with zero per-element Python cost. Kernels: `mul`
(one field multiply) and `horner16` (degree-16 polynomial, 16 mul + 16 add per
element). Run with `PYTHONPATH=frontend python3 benchmark/bench_batch.py`.

### GF(2^61 - 1) -- Mersenne61, fits galois's int64 JIT path

| kernel | ffjit | pure Python | galois (numpy/Numba) |
|---|---|---|---|
| mul | **0.69 ms (7 ns/elem)** | 10.6 ms (15.4x slower) | 13.3 ms (19.3x slower) |
| horner16 | **5.5 ms (55 ns/elem)** | 168 ms (30.5x slower) | 337 ms (61.1x slower) |

Even on a prime where galois has its best case (single-word, Numba-JITed
ufuncs), ffjit's fused kernel wins -- galois materializes a full array per
`*`/`+` while ffjit fuses the entire expression into one loop over the data.

### BN254 scalar field (254-bit prime) -- the cryptography gap

| kernel | ffjit | pure Python | galois |
|---|---|---|---|
| mul | **9.7 ms (97 ns/elem)** | 27.3 ms (2.8x slower) | field construction infeasible (>30 s) |
| horner16 | **40 ms (400 ns/elem)** | 420 ms (10.5x slower) | -- |

One-time costs at this size: ~36 ms to marshal 100k Python ints into a
`FieldArray`, ~300 ms to compile a kernel on first use (then disk-cached).
Chained `map` calls keep data in native buffers, so pipelines pay marshalling
only at the boundaries.

## Scalar kernels (Phase 1): one compiled call per evaluation

Workload: Horner evaluation of a degree-96 polynomial over the **BN254 scalar
field** (a 254-bit prime), 300 evaluations. Run with `python3 benchmark/bench_bn254.py`.

| Implementation | time / eval | vs pure Python | notes |
|---|---|---|---|
| pure Python (CPython big ints) | ~28 us | 1.0x | interpreter loop, `% p` per step |
| **ffjit (JIT, Montgomery)** | **~5.4 us** | **~5.2x faster** | compiled straight-line Montgomery arithmetic |
| `galois` | -- | -- | **could not construct GF(p) within 20 s** (must factor `p-1`); above 64 bits it also falls back to non-JIT Python object ufuncs |

All three agree on results where they run.

## Reading the numbers

- The ffjit win over pure Python comes from removing per-operation interpreter
  overhead and replacing trial division (`%`) with Montgomery reduction (a
  shift and conditional subtract). The gap widens with the amount of compute
  fused into a single compiled kernel.
- `galois` is the closest existing tool, but its Numba JIT only covers fields
  that fit in `int64`; for cryptographic 254-bit primes it cannot JIT at all,
  and even setting up the field object is expensive. This is precisely the gap
  ffjit targets.

## Current limitations (honest)

- Scalar kernel *bodies* are unrolled straight-line code, so very high-degree
  single-call kernels are slow to compile (~2 s at degree 64). The batch path
  sidesteps this for data-parallel work: the kernel is compiled once and the
  loop over N is in the generated code, so compile time is O(kernel), not O(N).
- Per-call `ctypes` marshalling (~a few microseconds) dominates for trivial
  one-operation *scalar* calls; use `f.map` over `FieldArray`s for
  data-parallel workloads, where marshalling is paid once at the boundary.
- `FieldArray` construction from Python ints is a Python-side O(N) loop
  (~360 ns/elem). Pipelines that stay in `FieldArray` form amortize it away.
