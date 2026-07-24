"""NTT, multi-output kernels, and Poly multiplication tests."""

import random

import pytest

import ffjit as ff
from ffjit.ntt import get_plan, root_of_unity, two_adicity

P_BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P_GOLDILOCKS = (1 << 64) - (1 << 32) + 1

NTT_PRIMES = [P_BN254, P_GOLDILOCKS]


# ---- multi-output kernels ----

@ff.jit
def bf(a, b, w):
    t = w * b
    return a + t, a - t


@pytest.mark.parametrize("p", NTT_PRIMES)
def test_multi_output_scalar(p):
    F = ff.GF(p)
    rng = random.Random(0)
    for _ in range(10):
        a, b, w = (rng.randrange(p) for _ in range(3))
        u, v = bf(F(a), F(b), F(w))
        assert int(u) == (a + w * b) % p
        assert int(v) == (a - w * b) % p


def test_multi_output_batch():
    p = P_BN254
    F = ff.GF(p)
    rng = random.Random(1)
    n = 100
    A = [rng.randrange(p) for _ in range(n)]
    B = [rng.randrange(p) for _ in range(n)]
    W = [rng.randrange(p) for _ in range(n)]
    U, V = bf.map(ff.FieldArray(F, A), ff.FieldArray(F, B), ff.FieldArray(F, W))
    assert U.to_ints() == [(a + w * b) % p for a, b, w in zip(A, B, W)]
    assert V.to_ints() == [(a - w * b) % p for a, b, w in zip(A, B, W)]


# ---- roots of unity ----

def test_two_adicity_known_values():
    assert two_adicity(P_BN254) == 28
    assert two_adicity(P_GOLDILOCKS) == 32
    assert two_adicity(65537) == 16


@pytest.mark.parametrize("p", NTT_PRIMES)
@pytest.mark.parametrize("k", [1, 4, 10])
def test_root_of_unity_has_exact_order(p, k):
    w = root_of_unity(p, k)
    assert pow(w, 1 << k, p) == 1
    assert pow(w, 1 << (k - 1), p) != 1


# ---- NTT ----

@pytest.mark.parametrize("p", NTT_PRIMES)
def test_ntt_matches_naive_dft(p):
    F = ff.GF(p)
    logn, n = 4, 16
    rng = random.Random(2)
    xs = [rng.randrange(p) for _ in range(n)]
    plan = get_plan(F, logn)
    X = plan.ntt(ff.FieldArray(F, xs)).to_ints()
    naive = [
        sum(xs[j] * pow(plan.w, j * k, p) for j in range(n)) % p
        for k in range(n)
    ]
    assert X == naive


@pytest.mark.parametrize("p", NTT_PRIMES)
@pytest.mark.parametrize("logn", [0, 1, 6, 11])
def test_ntt_roundtrip(p, logn):
    F = ff.GF(p)
    n = 1 << logn
    rng = random.Random(3)
    xs = [rng.randrange(p) for _ in range(n)]
    fa = ff.FieldArray(F, xs)
    assert ff.intt(ff.ntt(fa)).to_ints() == xs


def test_ntt_is_linear():
    p = P_BN254
    F = ff.GF(p)
    n = 64
    rng = random.Random(4)
    xs = [rng.randrange(p) for _ in range(n)]
    ys = [rng.randrange(p) for _ in range(n)]
    X = ff.ntt(ff.FieldArray(F, xs)).to_ints()
    Y = ff.ntt(ff.FieldArray(F, ys)).to_ints()
    Z = ff.ntt(ff.FieldArray(F, [(x + y) % p for x, y in zip(xs, ys)])).to_ints()
    assert Z == [(a + b) % p for a, b in zip(X, Y)]


def test_ntt_rejects_non_power_of_two():
    F = ff.GF(P_BN254)
    with pytest.raises(ValueError):
        ff.ntt(ff.FieldArray(F, [1, 2, 3]))


# ---- Poly ----

def _schoolbook(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return out


@pytest.mark.parametrize("p", NTT_PRIMES)
def test_poly_mul_small_uses_schoolbook(p):
    F = ff.GF(p)
    a = ff.Poly(F, [1, 2, 3])
    b = ff.Poly(F, [4, 5])
    assert (a * b).coeffs == _schoolbook([1, 2, 3], [4, 5], p)


@pytest.mark.parametrize("p", NTT_PRIMES)
def test_poly_mul_large_matches_schoolbook(p):
    F = ff.GF(p)
    rng = random.Random(5)
    a = [rng.randrange(p) for _ in range(150)]
    b = [rng.randrange(p) for _ in range(97)]
    pa, pb = ff.Poly(F, a), ff.Poly(F, b)
    prod = pa * pb
    assert prod.coeffs == _schoolbook(a, b, p)
    assert prod.degree == 150 + 97 - 2


def test_poly_ring_identities():
    p = P_BN254
    F = ff.GF(p)
    rng = random.Random(6)
    a = ff.Poly(F, [rng.randrange(p) for _ in range(80)])
    b = ff.Poly(F, [rng.randrange(p) for _ in range(80)])
    c = ff.Poly(F, [rng.randrange(p) for _ in range(80)])
    assert a * b == b * a
    assert a * (b + c) == a * b + a * c
    one = ff.Poly(F, [1])
    assert a * one == a


def test_poly_eval_agrees_with_mul():
    p = P_GOLDILOCKS
    F = ff.GF(p)
    rng = random.Random(7)
    a = ff.Poly(F, [rng.randrange(p) for _ in range(70)])
    b = ff.Poly(F, [rng.randrange(p) for _ in range(70)])
    x = rng.randrange(p)
    assert int((a * b)(x)) == int(a(x)) * int(b(x)) % p


# ---- negacyclic convolution ----

def _negacyclic_naive(a, b, p):
    """c_k = sum_{i+j = k} a_i b_j - sum_{i+j = n+k} a_i b_j (mod p)."""
    n = len(a)
    out = [0] * n
    for i in range(n):
        for j in range(n):
            k = i + j
            if k < n:
                out[k] = (out[k] + a[i] * b[j]) % p
            else:
                out[k - n] = (out[k - n] - a[i] * b[j]) % p
    return out


@pytest.mark.parametrize("p", NTT_PRIMES)
@pytest.mark.parametrize("logn", [0, 1, 4, 7])
def test_negacyclic_matches_naive(p, logn):
    F = ff.GF(p)
    n = 1 << logn
    rng = random.Random(8)
    a = [rng.randrange(p) for _ in range(n)]
    b = [rng.randrange(p) for _ in range(n)]
    got = ff.negacyclic_mul(ff.FieldArray(F, a), ff.FieldArray(F, b))
    assert got.to_ints() == _negacyclic_naive(a, b, p)


def test_negacyclic_x_to_n_is_minus_one():
    # x^(n/2) * x^(n/2) = x^n = -1 in GF(p)[x]/(x^n + 1)
    p = P_BN254
    F = ff.GF(p)
    n = 16
    half = [0] * n
    half[n // 2] = 1
    got = ff.negacyclic_mul(ff.FieldArray(F, half), ff.FieldArray(F, half))
    expected = [0] * n
    expected[0] = p - 1
    assert got.to_ints() == expected


def test_negacyclic_rejects_mismatched_operands():
    F = ff.GF(P_BN254)
    with pytest.raises(ValueError):
        ff.negacyclic_mul(ff.FieldArray(F, [1, 2]), ff.FieldArray(F, [1, 2, 3, 4]))
    with pytest.raises(ValueError):
        ff.negacyclic_mul(ff.FieldArray(F, [1, 2, 3]), ff.FieldArray(F, [1, 2, 3]))
