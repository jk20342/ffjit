"""Batched (``FieldArray`` / ``JittedFunction.map``) kernel tests.

Every batched result is checked elementwise against the pure-Python
``FieldVal`` reference arithmetic.
"""

import random

import pytest

import ffjit as ff

P_SMALL = 65537
P_BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617

PRIMES = [P_SMALL, (1 << 61) - 1, P_BN254]


@ff.jit
def k_mul(a, b):
    return a * b


@ff.jit
def k_affine(a, b, c):
    return a * b + c


@ff.jit
def k_invmul(a, b):
    return (a * b).inv()


@pytest.mark.parametrize("p", PRIMES)
def test_batch_mul_matches_reference(p):
    F = ff.GF(p)
    rng = random.Random(1234)
    n = 257  # deliberately not a power of two
    A = [F(rng.randrange(p)) for _ in range(n)]
    B = [F(rng.randrange(p)) for _ in range(n)]

    out = k_mul.map(ff.FieldArray(F, A), ff.FieldArray(F, B))
    assert isinstance(out, ff.FieldArray)
    assert len(out) == n
    assert out.to_ints() == [int(a * b) for a, b in zip(A, B)]


@pytest.mark.parametrize("p", PRIMES)
def test_batch_three_args(p):
    F = ff.GF(p)
    rng = random.Random(99)
    n = 64
    A = [F(rng.randrange(p)) for _ in range(n)]
    B = [F(rng.randrange(p)) for _ in range(n)]
    C = [F(rng.randrange(p)) for _ in range(n)]

    out = k_affine.map(ff.FieldArray(F, A), ff.FieldArray(F, B), ff.FieldArray(F, C))
    assert out.to_ints() == [int(a * b + c) for a, b, c in zip(A, B, C)]


@pytest.mark.parametrize("p", [P_SMALL, P_BN254])
def test_batch_with_inversion(p):
    F = ff.GF(p)
    rng = random.Random(7)
    n = 33
    A = [F(rng.randrange(1, p)) for _ in range(n)]
    B = [F(rng.randrange(1, p)) for _ in range(n)]

    out = k_invmul.map(ff.FieldArray(F, A), ff.FieldArray(F, B))
    assert out.to_ints() == [int((a * b).inv()) for a, b in zip(A, B)]


def test_batch_accepts_plain_sequences():
    F = ff.GF(P_SMALL)
    A = [F(3), F(5), F(65536)]
    B = [F(7), F(11), F(2)]
    out = k_mul.map(A, B)
    assert out.to_ints() == [int(a * b) for a, b in zip(A, B)]


def test_batch_chained_maps_stay_native():
    F = ff.GF(P_BN254)
    rng = random.Random(5)
    n = 50
    A = ff.FieldArray(F, [rng.randrange(P_BN254) for _ in range(n)])
    B = ff.FieldArray(F, [rng.randrange(P_BN254) for _ in range(n)])

    r = k_mul.map(A, B)
    r = k_mul.map(r, r)      # square
    r = k_affine.map(r, A, B)

    ai, bi = A.to_ints(), B.to_ints()
    expect = [
        ((pow((a * b) % P_BN254, 2, P_BN254) * a) + b) % P_BN254
        for a, b in zip(ai, bi)
    ]
    assert r.to_ints() == expect


def test_batch_length_mismatch_raises():
    F = ff.GF(P_SMALL)
    with pytest.raises(ValueError):
        k_mul.map(ff.FieldArray(F, [F(1), F(2)]), ff.FieldArray(F, [F(1)]))


def test_batch_scalar_and_map_agree():
    F = ff.GF(P_BN254)
    rng = random.Random(42)
    pairs = [(F(rng.randrange(P_BN254)), F(rng.randrange(P_BN254))) for _ in range(20)]
    scalar = [int(k_mul(a, b)) for a, b in pairs]
    batched = k_mul.map(
        ff.FieldArray(F, [a for a, _ in pairs]),
        ff.FieldArray(F, [b for _, b in pairs]),
    ).to_ints()
    assert scalar == batched


def test_fieldarray_roundtrip():
    F = ff.GF(P_BN254)
    vals = [0, 1, P_BN254 - 1, 123456789]
    fa = ff.FieldArray(F, vals)
    assert fa.to_ints() == vals
    assert [int(v) for v in fa.to_list()] == vals
