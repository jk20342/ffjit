//===- RuntimeCAPI.cpp - ffjit runtime C ABI implementation ---*- C++ -*-===//
//
// Implementation of the ffjit prime-field runtime. Multi-precision values
// are little-endian arrays of 64-bit limbs. Correctness over cleverness:
// everything here is straightforward schoolbook limb arithmetic.
//
//===--------------------------------------------------------------------===//

#include "ff/RuntimeCAPI.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

/// Maximum limb count supported without heap allocation. 8 limbs = 512 bits,
/// enough for all fields we currently target (e.g. BN254 needs 4).
constexpr size_t kMaxLimbs = 8;

//===--------------------------------------------------------------------===//
// Limb-vector helpers. All operate on little-endian arrays of `n` limbs.
//===--------------------------------------------------------------------===//

bool isZero(const uint64_t *a, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (a[i] != 0)
      return false;
  return true;
}

bool isOne(const uint64_t *a, size_t n) {
  if (a[0] != 1)
    return false;
  for (size_t i = 1; i < n; ++i)
    if (a[i] != 0)
      return false;
  return true;
}

bool isEven(const uint64_t *a) { return (a[0] & 1u) == 0; }

/// Returns <0, 0, >0 as a <=> b.
int cmp(const uint64_t *a, const uint64_t *b, size_t n) {
  for (size_t i = n; i-- > 0;) {
    if (a[i] < b[i])
      return -1;
    if (a[i] > b[i])
      return 1;
  }
  return 0;
}

/// r = a + b. Returns the carry out (0 or 1). r may alias a or b.
uint64_t add(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
  uint64_t carry = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned __int128 s = (unsigned __int128)a[i] + b[i] + carry;
    r[i] = (uint64_t)s;
    carry = (uint64_t)(s >> 64);
  }
  return carry;
}

/// r = a - b. Returns the borrow out (0 or 1). r may alias a or b.
uint64_t sub(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
  uint64_t borrow = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned __int128 d = (unsigned __int128)a[i] - b[i] - borrow;
    r[i] = (uint64_t)d;
    borrow = (uint64_t)((d >> 64) & 1);
  }
  return borrow;
}

/// a >>= 1, shifting in `topBit` (0 or 1) as the new most significant bit.
/// The extra top bit lets callers halve an (n*64 + 1)-bit value, which is
/// needed after computing x + p where both are n limbs wide.
void shr1(uint64_t *a, size_t n, uint64_t topBit) {
  for (size_t i = 0; i + 1 < n; ++i)
    a[i] = (a[i] >> 1) | (a[i + 1] << 63);
  a[n - 1] = (a[n - 1] >> 1) | (topBit << 63);
}

/// x = x / 2 mod p, assuming 0 <= x < p and p odd.
/// If x is even, halve directly; otherwise x + p is even (p odd), so halve
/// x + p, which is congruent to x/2 mod p and still < p after the shift.
void halveMod(uint64_t *x, const uint64_t *p, size_t n) {
  if (isEven(x)) {
    shr1(x, n, 0);
  } else {
    uint64_t carry = add(x, x, p, n);
    shr1(x, n, carry);
  }
}

/// x = (x - y) mod p, assuming 0 <= x, y < p.
void subMod(uint64_t *x, const uint64_t *y, const uint64_t *p, size_t n) {
  if (sub(x, x, y, n))
    add(x, x, p, n);
}

constexpr uint64_t kNoSlot = std::numeric_limits<uint64_t>::max();

uint32_t extractDigit(const uint64_t *scalar, size_t nlimbs, uint32_t bit,
                      uint32_t window) {
  const size_t limb = bit / 64;
  const uint32_t shift = bit % 64;
  if (limb >= nlimbs)
    return 0;
  uint64_t value = scalar[limb] >> shift;
  if (shift != 0 && shift + window > 64 && limb + 1 < nlimbs)
    value |= scalar[limb + 1] << (64 - shift);
  return static_cast<uint32_t>(value & ((uint64_t(1) << window) - 1));
}

struct ScheduleBuilder {
  explicit ScheduleBuilder(uint64_t inputs) : nextSlot(inputs) {}

  uint64_t add(uint64_t lhs, uint64_t rhs, uint32_t round) {
    const uint64_t out = nextSlot++;
    ops.push_back({FF_RT_POINT_ADD, round, lhs, rhs, out});
    return out;
  }

  uint64_t dbl(uint64_t lhs, uint32_t round) {
    const uint64_t out = nextSlot++;
    ops.push_back({FF_RT_POINT_DOUBLE, round, lhs, lhs, out});
    return out;
  }

  uint64_t reduce(std::vector<uint64_t> slots, uint32_t &round) {
    while (slots.size() > 1) {
      std::vector<uint64_t> next;
      next.reserve((slots.size() + 1) / 2);
      size_t i = 0;
      if (slots.size() & 1)
        next.push_back(slots[i++]);
      for (; i < slots.size(); i += 2)
        next.push_back(add(slots[i], slots[i + 1], round));
      slots.swap(next);
      ++round;
    }
    return slots.empty() ? kNoSlot : slots[0];
  }

  uint64_t nextSlot;
  std::vector<ff_rt_point_op> ops;
};

int32_t finishSchedule(const ScheduleBuilder &builder, uint32_t window,
                       uint32_t windows, uint64_t inputs, uint64_t result,
                       ff_rt_point_schedule *schedule, ff_rt_point_op *ops,
                       size_t capacity) {
  if (schedule == nullptr)
    return -1;
  schedule->version = FF_RT_MSM_SCHEDULE_VERSION;
  schedule->window = window;
  schedule->window_count = windows;
  schedule->reserved = 0;
  schedule->input_count = inputs;
  schedule->slot_count = builder.nextSlot;
  schedule->op_count = builder.ops.size();
  schedule->result_slot = result;
  if (ops == nullptr)
    return capacity == 0 ? 0 : -1;
  if (capacity < builder.ops.size())
    return -2;
  if (!builder.ops.empty())
    std::memcpy(ops, builder.ops.data(),
                builder.ops.size() * sizeof(ff_rt_point_op));
  return 0;
}

} // namespace

extern "C" {

int32_t ff_rt_abi_version(void) { return FF_RT_ABI_VERSION; }

// Binary extended Euclidean inversion (binary inversion algorithm), see
// Handbook of Applied Cryptography, Menezes/van Oorschot/Vanstone,
// Algorithm 14.61 with the simplification of Note 14.64 for computing a
// single inverse modulo an odd prime.
//
// Invariants maintained throughout (with p = modulus):
//   x1 * a == u (mod p)
//   x2 * a == v (mod p)
//   gcd(u, v) == gcd(a, p) == 1
//   0 <= x1, x2 < p
//
// Each step either halves an even u or v (adjusting x1/x2 by a modular
// halving, which handles the even-intermediate case by adding the odd
// modulus before the right shift), or subtracts the smaller of u, v from
// the larger. The loop terminates with u == 1 or v == 1, at which point
// the corresponding x is a^(-1) mod p.
void ff_rt_inv(uint64_t *out, const uint64_t *a, const uint64_t *modulus,
               size_t nlimbs) {
  if (nlimbs == 0 || nlimbs > kMaxLimbs) {
    std::fprintf(stderr, "ff_rt_inv: unsupported nlimbs=%zu (must be 1..%zu)\n",
                 nlimbs, kMaxLimbs);
    std::abort();
  }

  // Convention: inv(0) = 0.
  if (isZero(a, nlimbs)) {
    std::memset(out, 0, nlimbs * sizeof(uint64_t));
    return;
  }

  uint64_t u[kMaxLimbs], v[kMaxLimbs], x1[kMaxLimbs], x2[kMaxLimbs];
  const size_t n = nlimbs;

  std::memcpy(u, a, n * sizeof(uint64_t));       // u = a
  std::memcpy(v, modulus, n * sizeof(uint64_t)); // v = p
  std::memset(x1, 0, n * sizeof(uint64_t));      // x1 = 1
  x1[0] = 1;
  std::memset(x2, 0, n * sizeof(uint64_t)); // x2 = 0

  while (!isOne(u, n) && !isOne(v, n)) {
    while (!isZero(u, n) && isEven(u)) {
      shr1(u, n, 0);
      halveMod(x1, modulus, n);
    }
    while (!isZero(v, n) && isEven(v)) {
      shr1(v, n, 0);
      halveMod(x2, modulus, n);
    }
    // A zero remainder means gcd(a, modulus) != 1. Prime fields never take
    // this path, but returning zero keeps malformed composite inputs from
    // trapping the runtime in the even-value loops forever.
    if (isZero(u, n) || isZero(v, n)) {
      std::memset(out, 0, n * sizeof(uint64_t));
      return;
    }
    if (cmp(u, v, n) >= 0) {
      sub(u, u, v, n); // u -= v (no borrow: u >= v)
      subMod(x1, x2, modulus, n);
    } else {
      sub(v, v, u, n); // v -= u
      subMod(x2, x1, modulus, n);
    }
  }

  const uint64_t *result = isOne(u, n) ? x1 : x2;
  std::memcpy(out, result, n * sizeof(uint64_t));
}

int32_t ff_rt_msm_schedule(const uint64_t *scalars, size_t count,
                           size_t scalar_nlimbs, uint32_t scalar_bits,
                           uint32_t window, ff_rt_point_schedule *schedule,
                           ff_rt_point_op *ops, size_t op_capacity) {
  if (scalars == nullptr || schedule == nullptr || count == 0 ||
      scalar_nlimbs == 0 || scalar_bits == 0 || window == 0 || window > 16)
    return -1;

  const uint32_t windows = (scalar_bits + window - 1) / window;
  const uint32_t maxDigit = (uint32_t(1) << window) - 1;
  std::vector<std::vector<uint64_t>> buckets(size_t(windows) *
                                             (size_t(maxDigit) + 1));
  for (size_t point = 0; point < count; ++point) {
    const uint64_t *scalar = scalars + point * scalar_nlimbs;
    for (uint32_t w = 0; w < windows; ++w) {
      const uint32_t digit =
          extractDigit(scalar, scalar_nlimbs, w * window, window);
      if (digit != 0)
        buckets[size_t(w) * (maxDigit + 1) + digit].push_back(point);
    }
  }

  ScheduleBuilder builder(count);
  uint32_t round = 0;
  std::vector<uint64_t> reduced(buckets.size(), kNoSlot);
  bool pending = true;
  while (pending) {
    pending = false;
    bool emitted = false;
    for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
      auto &slots = buckets[bucket];
      if (slots.size() <= 1)
        continue;
      pending = true;
      emitted = true;
      std::vector<uint64_t> next;
      next.reserve((slots.size() + 1) / 2);
      size_t i = 0;
      if (slots.size() & 1)
        next.push_back(slots[i++]);
      for (; i < slots.size(); i += 2)
        next.push_back(builder.add(slots[i], slots[i + 1], round));
      slots.swap(next);
    }
    if (emitted)
      ++round;
  }
  for (size_t bucket = 0; bucket < buckets.size(); ++bucket)
    if (!buckets[bucket].empty())
      reduced[bucket] = buckets[bucket][0];

  std::vector<uint64_t> running(windows, kNoSlot);
  std::vector<uint64_t> accum(windows, kNoSlot);
  for (uint32_t digit = maxDigit; digit != 0; --digit) {
    bool emitted = false;
    for (uint32_t w = 0; w < windows; ++w) {
      const uint64_t bucket = reduced[size_t(w) * (maxDigit + 1) + digit];
      if (bucket == kNoSlot)
        continue;
      if (running[w] == kNoSlot)
        running[w] = bucket;
      else {
        running[w] = builder.add(running[w], bucket, round);
        emitted = true;
      }
    }
    if (emitted)
      ++round;
    emitted = false;
    for (uint32_t w = 0; w < windows; ++w) {
      if (running[w] == kNoSlot)
        continue;
      if (accum[w] == kNoSlot)
        accum[w] = running[w];
      else {
        accum[w] = builder.add(accum[w], running[w], round);
        emitted = true;
      }
    }
    if (emitted)
      ++round;
  }

  uint64_t result = kNoSlot;
  for (uint32_t wi = windows; wi-- > 0;) {
    if (result != kNoSlot) {
      for (uint32_t bit = 0; bit < window; ++bit)
        result = builder.dbl(result, round++);
    }
    if (accum[wi] != kNoSlot) {
      if (result == kNoSlot)
        result = accum[wi];
      else
        result = builder.add(result, accum[wi], round++);
    }
  }
  return finishSchedule(builder, window, windows, count, result, schedule, ops,
                        op_capacity);
}

int32_t ff_rt_fixed_base_schedule(const uint64_t *scalar, size_t scalar_nlimbs,
                                  uint32_t scalar_bits, uint32_t window,
                                  uint32_t window_count,
                                  ff_rt_point_schedule *schedule,
                                  ff_rt_point_op *ops, size_t op_capacity) {
  if (scalar == nullptr || schedule == nullptr || scalar_nlimbs == 0 ||
      scalar_bits == 0 || window == 0 || window > 16 || window_count == 0)
    return -1;
  const uint64_t stride = (uint64_t(1) << window) - 1;
  const uint64_t inputs = stride * window_count;
  std::vector<uint64_t> selected;
  selected.reserve(window_count);
  for (uint32_t w = 0; w < window_count; ++w) {
    const uint32_t digit =
        extractDigit(scalar, scalar_nlimbs, w * window, window);
    if (digit != 0)
      selected.push_back(uint64_t(w) * stride + digit - 1);
  }
  ScheduleBuilder builder(inputs);
  uint32_t round = 0;
  const uint64_t result = builder.reduce(std::move(selected), round);
  return finishSchedule(builder, window, window_count, inputs, result, schedule,
                        ops, op_capacity);
}

void ff_rt_dump_limbs(const char *label, const uint64_t *a, size_t nlimbs) {
  std::fprintf(stderr, "%s = 0x", label ? label : "");
  if (nlimbs == 0) {
    std::fprintf(stderr, "0\n");
    return;
  }
  // Print most significant limb first so the output reads as one number.
  for (size_t i = nlimbs; i-- > 0;)
    std::fprintf(stderr, "%016llx", (unsigned long long)a[i]);
  std::fprintf(stderr, "\n");
}

} // extern "C"
