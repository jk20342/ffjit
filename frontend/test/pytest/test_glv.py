"""GLV endomorphism tests: constants, lattice decomposition, scalar mult."""

import random

import pytest

import ffjit as ff
from ffjit.curve import _glv_decompose, _round_div

GROUPS = {
    "bn254_g1": ff.bn254_g1,
    "secp256k1": ff.secp256k1,
}


@pytest.fixture(scope="module", params=list(GROUPS))
def group(request):
    return GROUPS[request.param]()


def test_round_div():
    assert _round_div(7, 2) == 4
    assert _round_div(-7, 2) == -4
    assert _round_div(6, 3) == 2
    assert _round_div(-6, 3) == -2
    assert _round_div(0, 5) == 0


def test_endomorphism_constants(group):
    curve, G, r = group
    glv = curve._glv
    lam, beta = glv["lam"], glv["beta"]
    assert (lam * lam + lam + 1) % r == 0
    assert pow(beta, 3, curve.q) == 1 and beta != 1
    # phi acts as multiplication by lambda on the subgroup
    assert curve._endo(G) == G._plain_mul(lam)


def test_short_basis_vectors_in_lattice(group):
    curve, G, r = group
    glv = curve._glv
    for (a, b) in (glv["v1"], glv["v2"]):
        assert (a + b * glv["lam"]) % r == 0
        assert abs(a).bit_length() <= r.bit_length() // 2 + 2
        assert abs(b).bit_length() <= r.bit_length() // 2 + 2


def test_decomposition_invariants(group):
    curve, G, r = group
    glv = curve._glv
    rng = random.Random(0)
    half = r.bit_length() // 2 + 2
    for _ in range(300):
        k = rng.randrange(r)
        k1, k2 = _glv_decompose(k, glv)
        assert (k1 + k2 * glv["lam"] - k) % r == 0
        assert abs(k1).bit_length() <= half
        assert abs(k2).bit_length() <= half


def test_glv_mul_matches_plain(group):
    curve, G, r = group
    rng = random.Random(1)
    P = rng.randrange(2, r) * G
    for _ in range(5):
        k = rng.randrange(r)
        assert P._glv_mul(k) == P._plain_mul(k)
    # edge scalars
    for k in (1, 2, r - 1, r // 2):
        assert P._glv_mul(k) == P._plain_mul(k)


def test_rmul_dispatches_to_glv(group):
    curve, G, r = group
    # __rmul__ must be correct whichever path it takes, including
    # reduction of scalars >= r
    rng = random.Random(2)
    k = rng.randrange(r)
    assert (k + r) * G == k * G
    assert (r - 1) * G == -G


def test_msm_with_glv_matches_naive(group):
    curve, G, r = group
    rng = random.Random(3)
    n = 25
    pts = [rng.randrange(1, r) * G for _ in range(n)]
    ks = [rng.randrange(r) for _ in range(n)]
    naive = curve.infinity()
    for P, k in zip(pts, ks):
        naive = naive + k * P
    assert ff.msm(pts, ks) == naive


def test_glv_disabled_curve_still_works():
    # a manually constructed curve has no GLV data; everything falls back
    q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    curve = ff.Curve(ff.GF(q), 0, 3)
    assert curve._glv is None
    G = curve.point(1, 2)
    assert 5 * G == G + G + G + G + G
