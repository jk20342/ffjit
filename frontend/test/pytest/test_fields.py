"""End-to-end correctness tests for the ffjit compiler.

Each JIT-compiled kernel is checked against the pure-Python ``FieldVal``
reference (CPython big integers), including property-based tests over the full
range of several primes of increasing size.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ffjit as ff

# A ladder of primes exercising 1, 2, and 4 limb widths.
PRIMES = {
    "gf17bit": 65537,                       # 17 bits  -> W=64
    "mersenne61": (1 << 61) - 1,            # 61 bits  -> W=64
    "prime128": (1 << 127) - 1,             # 127 bits -> W=128 (Mersenne)
    "bn254": 21888242871839275222246405745257275088548364400416034343698204186575808495617,
    "bls12_381": 52435875175126190479447740508185965837690552500527637822603658699938581184513,
}


@pytest.mark.parametrize("pname", list(PRIMES))
def test_arith_smoke(pname):
    p = PRIMES[pname]
    F = ff.GF(p)

    @ff.jit
    def k(x, y):
        return (x * y + x - y)

    for x, y in [(0, 0), (1, 0), (0, 1), (2, 3), (p - 1, p - 1)]:
        got = k(F(x), F(y))
        assert int(got) == (x * y + x - y) % p


@pytest.mark.parametrize("pname", list(PRIMES))
def test_inverse(pname):
    p = PRIMES[pname]
    F = ff.GF(p)

    @ff.jit
    def inv(x):
        return x.inv()

    assert int(inv(F(0))) == 0            # convention
    assert int(inv(F(1))) == 1
    for x in [2, 3, p - 1, p // 2, 123456789 % p]:
        r = int(inv(F(x)))
        assert (x * r) % p == 1


def _hyp_field(p):
    F = ff.GF(p)

    @ff.jit
    def poly(x, a, b, c):
        # a*x^3 + b*x^2 + c  exercised through add/sub/mul chains
        return a * x * x * x + b * (x * x) + c

    def ref(x, a, b, c):
        return (a * x**3 + b * x**2 + c) % p

    return F, poly, ref


@settings(max_examples=200, deadline=None)
@given(
    x=st.integers(min_value=0),
    a=st.integers(min_value=0),
    b=st.integers(min_value=0),
    c=st.integers(min_value=0),
)
def test_bn254_polynomial_property(x, a, b, c):
    p = PRIMES["bn254"]
    F, poly, ref = _hyp_field(p)
    x, a, b, c = x % p, a % p, b % p, c % p
    got = poly(F(x), F(a), F(b), F(c))
    assert int(got) == ref(x, a, b, c)


@settings(max_examples=200, deadline=None)
@given(v=st.integers(min_value=1))
def test_bn254_inverse_property(v):
    p = PRIMES["bn254"]
    F = ff.GF(p)

    @ff.jit
    def inv(x):
        return x.inv()

    v = v % p
    if v == 0:
        return
    r = int(inv(F(v)))
    assert (v * r) % p == 1


def test_naive_reduction_matches_montgomery():
    """The urem-based oracle path must agree with the Montgomery path."""
    p = PRIMES["bn254"]
    F = ff.GF(p)

    @ff.jit(montgomery=True)
    def km(x, y):
        return x * y + x

    @ff.jit(montgomery=False)
    def kn(x, y):
        return x * y + x

    for x, y in [(2, 3), (p - 1, 7), (12345, 67890)]:
        assert int(km(F(x), F(y))) == int(kn(F(x), F(y))) == (x * y + x) % p


def test_division_operator():
    p = PRIMES["bn254"]
    F = ff.GF(p)

    @ff.jit
    def d(x, y):
        return x / y

    x, y = 100, 7
    got = int(d(F(x), F(y)))
    assert (got * y) % p == x % p
