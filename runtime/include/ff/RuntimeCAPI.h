//===- RuntimeCAPI.h - ffjit runtime C ABI --------------------*- C -*-===//
//
// Stable C ABI for the optional ffjit prime-field runtime. The current
// inversion lowering is pure IR; these symbols support runtime-backed
// kernels and independent testing.
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

/// Current ABI version of this runtime.
#define FF_RT_ABI_VERSION 3

/// Returns FF_RT_ABI_VERSION.
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
/// Convention: if a == 0 or is not invertible, out is set to 0.
///
/// `out` may alias `a`; it must not alias `modulus`.
void ff_rt_inv(uint64_t *out, const uint64_t *a, const uint64_t *modulus,
               size_t nlimbs);

/// Version of the compact point-operation schedule format.
#define FF_RT_MSM_SCHEDULE_VERSION 1

enum ff_rt_point_op_kind {
  FF_RT_POINT_ADD = 1,
  FF_RT_POINT_DOUBLE = 2,
};

/// One operation in a dependency-ordered point schedule.
///
/// Slots [0, input_count) refer to caller-owned input points. Each operation
/// writes one new slot. Operations with equal `round` have no dependencies on
/// each other and may be submitted in one native batch call.
typedef struct ff_rt_point_op {
  uint32_t kind;
  uint32_t round;
  uint64_t lhs;
  uint64_t rhs;
  uint64_t out;
} ff_rt_point_op;

/// Metadata for an MSM or fixed-base schedule.
typedef struct ff_rt_point_schedule {
  uint32_t version;
  uint32_t window;
  uint32_t window_count;
  uint32_t reserved;
  uint64_t input_count;
  uint64_t slot_count;
  uint64_t op_count;
  uint64_t result_slot;
} ff_rt_point_schedule;

/// Build a Pippenger point-operation schedule.
///
/// Scalars are `count` rows of `scalar_nlimbs` little-endian limbs. The first
/// call may pass `ops = NULL, op_capacity = 0` to query `op_count`; a second
/// call supplies that many operations. Returns 0 on success, -1 for invalid
/// arguments, or -2 when `op_capacity` is too small.
int32_t ff_rt_msm_schedule(const uint64_t *scalars, size_t count,
                           size_t scalar_nlimbs, uint32_t scalar_bits,
                           uint32_t window, ff_rt_point_schedule *schedule,
                           ff_rt_point_op *ops, size_t op_capacity);

/// Build a balanced-addition schedule for one fixed-base comb lookup.
///
/// Input slots use flattened table order
/// `window_index * ((1 << window) - 1) + digit - 1`.
int32_t ff_rt_fixed_base_schedule(const uint64_t *scalar, size_t scalar_nlimbs,
                                  uint32_t scalar_bits, uint32_t window,
                                  uint32_t window_count,
                                  ff_rt_point_schedule *schedule,
                                  ff_rt_point_op *ops, size_t op_capacity);

/// Debug helper: prints `label` followed by the value of `a` (little-endian
/// limbs, printed as one big-endian hex number) to stderr.
void ff_rt_dump_limbs(const char *label, const uint64_t *a, size_t nlimbs);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FFJIT_RUNTIME_C_API_H
