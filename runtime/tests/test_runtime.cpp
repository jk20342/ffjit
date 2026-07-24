//===- test_runtime.cpp - assert-based tests for the ffjit runtime -------===//
//
// No test framework dependency: plain asserts. Verification arithmetic
// (bigint multiply and mod) is implemented here in the simplest possible
// way, independently of the runtime code under test.
//
//===--------------------------------------------------------------------===//

#include "ff/RuntimeCAPI.h"

// These tests are assert-based; make sure asserts survive Release builds.
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

//===--------------------------------------------------------------------===//
// Single-limb tests: verify a * inv(a) mod p == 1 with __int128.
//===--------------------------------------------------------------------===//

void testSingleLimbPrime(uint64_t p) {
  const uint64_t values[] = {1,          2,          3,     65536, 12345,
                             0xdeadbeef, 0xffffffff, p - 1, p / 2, p / 3 + 1};
  for (uint64_t a : values) {
    a %= p;
    if (a == 0)
      continue;
    uint64_t inv = 0;
    ff_rt_inv(&inv, &a, &p, 1);
    assert(inv < p && "inverse must be reduced");
    unsigned __int128 prod = (unsigned __int128)a * inv;
    assert((uint64_t)(prod % p) == 1 && "a * inv(a) mod p must be 1");
  }
}

//===--------------------------------------------------------------------===//
// Multi-limb verification helpers (test-only, schoolbook).
//===--------------------------------------------------------------------===//

// r[0..2n) = a[0..n) * b[0..n), schoolbook.
void mulFull(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
  std::memset(r, 0, 2 * n * sizeof(uint64_t));
  for (size_t i = 0; i < n; ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < n; ++j) {
      unsigned __int128 t = (unsigned __int128)a[i] * b[j] + r[i + j] + carry;
      r[i + j] = (uint64_t)t;
      carry = (uint64_t)(t >> 64);
    }
    r[i + n] = carry;
  }
}

int cmpN(const uint64_t *a, const uint64_t *b, size_t n) {
  for (size_t i = n; i-- > 0;) {
    if (a[i] < b[i])
      return -1;
    if (a[i] > b[i])
      return 1;
  }
  return 0;
}

// a -= b, returns borrow.
uint64_t subN(uint64_t *a, const uint64_t *b, size_t n) {
  uint64_t borrow = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned __int128 d = (unsigned __int128)a[i] - b[i] - borrow;
    a[i] = (uint64_t)d;
    borrow = (uint64_t)((d >> 64) & 1);
  }
  return borrow;
}

// rem[0..n) = x[0..xn) mod p[0..n), via bit-by-bit shift-subtract long
// division. Slow but obviously correct; fine for test code.
void modN(uint64_t *rem, const uint64_t *x, size_t xn, const uint64_t *p,
          size_t n) {
  std::memset(rem, 0, n * sizeof(uint64_t));
  for (size_t bit = xn * 64; bit-- > 0;) {
    // rem = rem * 2 + bit(x, bit)
    uint64_t carryOut = rem[n - 1] >> 63;
    for (size_t i = n; i-- > 1;)
      rem[i] = (rem[i] << 1) | (rem[i - 1] >> 63);
    rem[0] = (rem[0] << 1) | ((x[bit / 64] >> (bit % 64)) & 1);
    if (carryOut || cmpN(rem, p, n) >= 0)
      subN(rem, p, n);
  }
}

bool isOneN(const uint64_t *a, size_t n) {
  if (a[0] != 1)
    return false;
  for (size_t i = 1; i < n; ++i)
    if (a[i] != 0)
      return false;
  return true;
}

// Checks a * ff_rt_inv(a) == 1 (mod p) for an n-limb value.
void checkInverse(const uint64_t *a, const uint64_t *p, size_t n) {
  uint64_t inv[8], prod[16], rem[8];
  ff_rt_inv(inv, a, p, n);
  assert(cmpN(inv, p, n) < 0 && "inverse must be reduced");
  mulFull(prod, a, inv, n);
  modN(rem, prod, 2 * n, p, n);
  assert(isOneN(rem, n) && "a * inv(a) mod p must be 1");
}

//===--------------------------------------------------------------------===//
// BN254 scalar field tests.
//===--------------------------------------------------------------------===//

void testBN254() {
  // p =
  // 21888242871839275222246405745257275088548364400416034343698204186575808495617
  const uint64_t p[4] = {0x43E1F593F0000001ULL, 0x2833E84879B97091ULL,
                         0xB85045B68181585DULL, 0x30644E72E131A029ULL};

  const uint64_t one[4] = {1, 0, 0, 0};
  const uint64_t two[4] = {2, 0, 0, 0};
  const uint64_t three[4] = {3, 0, 0, 0};
  // A fixed pseudorandom value < p (top limb below p's top limb).
  const uint64_t big[4] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL,
                           0x0F1E2D3C4B5A6978ULL, 0x2C4D5E6F708192A3ULL};
  const uint64_t pMinus1[4] = {0x43E1F593F0000000ULL, 0x2833E84879B97091ULL,
                               0xB85045B68181585DULL, 0x30644E72E131A029ULL};

  checkInverse(one, p, 4);
  checkInverse(two, p, 4);
  checkInverse(three, p, 4);
  checkInverse(big, p, 4);
  checkInverse(pMinus1, p, 4);

  // inv(1) must be exactly 1, and inv(p-1) must be p-1 (self-inverse).
  uint64_t inv[4];
  ff_rt_inv(inv, one, p, 4);
  assert(std::memcmp(inv, one, sizeof(one)) == 0);
  ff_rt_inv(inv, pMinus1, p, 4);
  assert(std::memcmp(inv, pMinus1, sizeof(pMinus1)) == 0);
}

void testZeroConvention() {
  const uint64_t p1 = 65537;
  uint64_t zero1 = 0, out1 = 0xffffffffffffffffULL;
  ff_rt_inv(&out1, &zero1, &p1, 1);
  assert(out1 == 0 && "inv(0) must be 0 (1 limb)");

  const uint64_t p[4] = {0x43E1F593F0000001ULL, 0x2833E84879B97091ULL,
                         0xB85045B68181585DULL, 0x30644E72E131A029ULL};
  const uint64_t zero[4] = {0, 0, 0, 0};
  uint64_t out[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
  ff_rt_inv(out, zero, p, 4);
  for (int i = 0; i < 4; ++i)
    assert(out[i] == 0 && "inv(0) must be 0 (4 limbs)");
}

} // namespace

int main() {
  assert(ff_rt_abi_version() == 1);

  testSingleLimbPrime(65537);                  // Fermat prime F4
  testSingleLimbPrime(2305843009213693951ULL); // Mersenne prime 2^61 - 1
  testBN254();
  testZeroConvention();

  std::printf("ALL RUNTIME TESTS PASSED\n");
  return 0;
}
