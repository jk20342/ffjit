ffjit runtime
=============

A small C++17 runtime library for the ffjit prime-field JIT compiler. It
exposes a **stable C ABI** (``include/ff/RuntimeCAPI.h``) that JIT-compiled
code links against for operations that are impractical to inline, currently:

- ``ff_rt_abi_version`` -- ABI version query (returns 1).
- ``ff_rt_inv`` -- modular inverse modulo an odd prime, for values given as
  little-endian arrays of 64-bit limbs (up to 8 limbs / 512 bits). Uses the
  binary extended Euclidean algorithm (Handbook of Applied Cryptography,
  Algorithm 14.61 / binary inversion). Convention: ``inv(0) = 0``.
- ``ff_rt_dump_limbs`` -- debug hex dump of a limb array to stderr.

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

This produces the static library ``runtime/build/libff_rt.a`` (built with
``-fPIC`` so it can be linked into shared objects or JIT sessions) and runs
the assert-based test suite ``test_runtime``.
