//===- RuntimeCAPI.h - ffjit runtime C ABI --------------------*- C -*-===//
//
// Stable C ABI for the ffjit prime-field runtime. JIT-compiled code calls
// these symbols for operations that are not worth inlining (currently
// modular inversion) and for debugging helpers.
//
// ABI stability: the signatures in this header form a versioned contract.
// Any incompatible change must bump the value returned by
// ff_rt_abi_version().
//
//===--------------------------------------------------------------------===//

#ifndef FFJIT_RUNTIME_C_API_H
#define FFJIT_RUNTIME_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Current ABI version of this runtime. Returns 1.
int32_t ff_rt_abi_version(void);

/// Modular inverse in a prime field.
///
/// Computes out = a^(-1) mod modulus, using the binary extended Euclidean
/// algorithm. All values are little-endian arrays of `nlimbs` 64-bit limbs
/// (limb 0 is least significant).
///
/// Preconditions:
///   - `modulus` is an odd prime
///   - 0 <= a < modulus
///   - 1 <= nlimbs <= 8 (the runtime aborts on larger sizes for now)
///
/// Convention: if a == 0, out is set to 0.
///
/// `out` may alias `a`; it must not alias `modulus`.
void ff_rt_inv(uint64_t *out, const uint64_t *a, const uint64_t *modulus,
               size_t nlimbs);

/// Debug helper: prints `label` followed by the value of `a` (little-endian
/// limbs, printed as one big-endian hex number) to stderr.
void ff_rt_dump_limbs(const char *label, const uint64_t *a, size_t nlimbs);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FFJIT_RUNTIME_C_API_H
