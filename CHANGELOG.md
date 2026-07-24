# Changelog

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
