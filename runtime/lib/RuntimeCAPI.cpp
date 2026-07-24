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

} // namespace

extern "C" {

int32_t ff_rt_abi_version(void) { return 1; }

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
    while (isEven(u)) {
      shr1(u, n, 0);
      halveMod(x1, modulus, n);
    }
    while (isEven(v)) {
      shr1(v, n, 0);
      halveMod(x2, modulus, n);
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
