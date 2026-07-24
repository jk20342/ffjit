"""Elliptic-curve tests: Jacobian kernels, group laws, scalar mult, MSM.

Everything is cross-checked against an independent pure-Python affine
implementation (inversion via ``pow(x, -1, q)``).
"""

import random

import pytest

import ffjit as ff

CURVES = {
    "bn254_g1": ff.bn254_g1,
    "secp256k1": ff.secp256k1,
}


@pytest.fixture(scope="module", params=list(CURVES))
def group(request):
    return CURVES[request.param]()


# ---- independent affine reference ----

class AffineRef:
    def __init__(self, curve):
        self.q, self.a = curve.q, curve.a

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        (x1, y1), (x2, y2) = P, Q
        q = self.q
        if x1 == x2 and (y1 + y2) % q == 0:
            return None
        if P == Q:
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, q) % q
        else:
            lam = (y2 - y1) * pow(x2 - x1, -1, q) % q
        x3 = (lam * lam - x1 - x2) % q
        return (x3, (lam * (x1 - x3) - y1) % q)

    def mul(self, k, P):
        acc = None
        for bit in bin(k)[2:]:
            acc = self.add(acc, acc)
            if bit == "1":
                acc = self.add(acc, P)
        return acc


# ---- construction ----

def test_point_validation(group):
    curve, G, r = group
    with pytest.raises(ValueError):
        curve.point(1, 1)


def test_generator_on_curve(group):
    curve, G, r = group
    x, y = G.to_affine()
    assert (y * y - (x**3 + curve.a * x + curve.b)) % curve.q == 0


# ---- group laws ----

def test_identity_and_inverse(group):
    curve, G, r = group
    inf = curve.infinity()
    assert (G + inf) == G
    assert (inf + G) == G
    assert (G - G).is_infinity
    assert (inf + inf).is_infinity
    assert (-inf).is_infinity


def test_double_consistency(group):
    curve, G, r = group
    assert G.double() == G + G          # + falls into the H=0 slow path
    assert G.double().double() == 4 * G


def test_commutativity_associativity(group):
    curve, G, r = group
    rng = random.Random(1)
    P, Q, R = (rng.randrange(2, r) * G for _ in range(3))
    assert P + Q == Q + P
    assert (P + Q) + R == P + (Q + R)


def test_order_annihilates_generator(group):
    curve, G, r = group
    assert (r * G).is_infinity
    assert ((r - 1) * G) == -G


# ---- scalar multiplication ----

def test_scalar_mul_matches_affine_reference(group):
    curve, G, r = group
    ref = AffineRef(curve)
    g_aff = G.to_affine()
    rng = random.Random(2)
    for _ in range(5):
        k = rng.randrange(1, r)
        assert (k * G).to_affine() == ref.mul(k, g_aff)


def test_scalar_mul_edge_cases(group):
    curve, G, r = group
    assert (0 * G).is_infinity
    assert 1 * G == G
    assert (-3) * G == -(3 * G)
    assert 2 * curve.infinity() == curve.infinity()


def test_scalar_distributes(group):
    curve, G, r = group
    rng = random.Random(3)
    a, b = rng.randrange(r), rng.randrange(r)
    assert ((a + b) % r) * G == a * G + b * G
    assert (a * b % r) * G == a * (b * G)


# ---- MSM ----

def test_msm_matches_naive(group):
    curve, G, r = group
    rng = random.Random(4)
    n = 30
    pts = [rng.randrange(1, r) * G for _ in range(n)]
    ks = [rng.randrange(r) for _ in range(n)]
    naive = curve.infinity()
    for P, k in zip(pts, ks):
        naive = naive + k * P
    assert ff.msm(pts, ks) == naive


def test_msm_various_windows():
    curve, G, r = ff.bn254_g1()
    rng = random.Random(5)
    n = 20
    pts = [rng.randrange(1, r) * G for _ in range(n)]
    ks = [rng.randrange(r) for _ in range(n)]
    expected = ff.msm(pts, ks)
    for c in (2, 5, 13):
        assert ff.msm(pts, ks, window=c) == expected


def test_msm_edge_cases():
    curve, G, r = ff.bn254_g1()
    inf = curve.infinity()
    # zero scalars, negative scalars, points at infinity, duplicates
    assert ff.msm([G], [0]).is_infinity
    assert ff.msm([G, G], [3, -3]).is_infinity
    assert ff.msm([G, inf], [5, 7]) == 5 * G
    assert ff.msm([G, G, G], [1, 1, 1]) == 3 * G
    assert ff.msm([G], [r]).is_infinity


def test_msm_single_point():
    curve, G, r = ff.bn254_g1()
    k = 0xDEADBEEFCAFEBABE
    assert ff.msm([G], [k]) == k * G
