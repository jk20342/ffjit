ffjit runtime
=============

A small C++17 runtime library for the ffjit prime-field JIT compiler. It
exposes a **stable C ABI** (``include/ff/RuntimeCAPI.h``) for optional
runtime-backed kernels and independent testing:

- ``ff_rt_abi_version`` -- ABI version query (returns 3).
- ``ff_rt_inv`` -- modular inverse modulo an odd prime, for values given as
  little-endian arrays of 64-bit limbs (up to 8 limbs / 512 bits). Uses the
  binary extended Euclidean algorithm (Handbook of Applied Cryptography,
  Algorithm 14.61 / binary inversion). Convention: ``inv(0) = 0``.
- ``ff_rt_dump_limbs`` -- debug hex dump of a limb array to stderr.
- ``ff_rt_msm_schedule`` -- scalar digit extraction plus Pippenger bucket,
  reduction, aggregation, and cross-window point-operation scheduling.
- ``ff_rt_fixed_base_schedule`` -- fixed-base comb digit extraction and
  balanced repeated-addition scheduling.

The scheduling APIs return compact, versioned operation arrays. Python retains
ownership of curve points and submits each independent schedule round to the
generated point batch kernels. This avoids unsafe cross-DSO callback ownership
while moving control-heavy scalar and bucket planning into the C runtime.

The frontend uses ``FFJIT_NATIVE_MSM=0`` for the Python reference scheduler and
``FFJIT_NATIVE_MSM=strict`` to reject runtime fallback. Generated one-call
batch inversion has matching ``FFJIT_NATIVE_BATCH_INV`` controls.

The current default inversion lowering is a self-contained pure-IR Fermat
``scf.for`` loop. ``@ff.jit(inv="runtime")`` selects the runtime-backed XGCD
lowering instead.

ABI stability
-------------

The signatures in ``RuntimeCAPI.h`` are a versioned contract between the
compiler and the runtime. Any incompatible change must bump the value
returned by ``ff_rt_abi_version()``. Symbols use plain C linkage.

Build and test
--------------

From the repository root::

    cmake -G Ninja -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
    cmake --build runtime/build
    ctest --test-dir runtime/build --output-on-failure

This produces ``runtime/build/libff_rt.a`` and
``runtime/build/libff_rt.so``. The static library is built with ``-fPIC``;
the shared library is loaded by Python and linked by runtime-backed JIT
kernels. The final command runs the assert-based ``test_runtime`` suite.
