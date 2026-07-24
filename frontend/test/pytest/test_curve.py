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
    "bls12_381_g1": ff.bls12_381_g1,  # 381-bit: exercises the 7-limb path
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


def test_msm_glv_native_reference_parity(group, monkeypatch):
    from ffjit.curve import _msm_ref
    curve, G, r = group
    points = [i * G for i in range(1, 9)]
    scalars = [(i * i * 0x12345) % r for i in range(1, 9)]
    expected = _msm_ref(points, scalars, window=3)
    monkeypatch.setenv("FFJIT_NATIVE_MSM", "strict")
    for window in (2, 3, 5):
        assert ff.msm(points, scalars, window=window) == expected


def test_msm_deterministic_128(monkeypatch):
    curve, G, r = ff.bn254_g1()
    points = [i * G for i in range(1, 129)]
    scalars = [((i * 0x9E3779B1) ^ (i << 17)) % r for i in range(128)]
    expected_scalar = sum((i + 1) * k for i, k in enumerate(scalars)) % r
    monkeypatch.setenv("FFJIT_NATIVE_MSM", "strict")
    assert ff.msm(points, scalars, window=5) == expected_scalar * G


def test_msm_environment_fallback_and_strict(monkeypatch):
    import ffjit.curve as curve_module
    curve, G, r = ff.bn254_g1()

    def unavailable():
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(curve_module, "get_runtime", unavailable)
    monkeypatch.setenv("FFJIT_NATIVE_MSM", "1")
    assert ff.msm([G, -G], [7, 2]) == 5 * G
    monkeypatch.setenv("FFJIT_NATIVE_MSM", "strict")
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        ff.msm([G], [1])


# ---- batch-affine internals ----

def test_batch_inv():
    from ffjit.curve import _batch_inv
    q = ff.bn254_g1()[0].q
    rng = random.Random(6)
    xs = [rng.randrange(1, q) for _ in range(37)]
    assert all(x * y % q == 1 for x, y in zip(xs, _batch_inv(q, xs)))


def test_generated_batch_inv_preserves_zero(monkeypatch):
    from ffjit.array import FieldArray
    from ffjit.curve import _batch_inv_array
    curve, G, r = ff.bn254_g1()
    monkeypatch.setenv("FFJIT_NATIVE_BATCH_INV", "strict")
    xs = [0, 1, 2, 0, curve.q - 1, 7]
    got = _batch_inv_array(FieldArray(curve.field, xs)).to_ints()
    assert got[0] == got[3] == 0
    assert all(
        inverse == (0 if value == 0 else pow(value, -1, curve.q))
        for value, inverse in zip(xs, got)
    )


def test_batch_affine_add_exceptional_cases():
    from ffjit.curve import _batch_affine_add
    curve, G, r = ff.bn254_g1()
    P2, P3 = (2 * G).to_affine(), (3 * G).to_affine()
    neg2 = (-(2 * G)).to_affine()
    # generic add, doubling (x1 == x2, y1 == y2), annihilation (P + -P)
    out = _batch_affine_add(curve, [P2, P2, P2], [P3, P2, neg2])
    assert out[0] == (5 * G).to_affine()
    assert out[1] == (4 * G).to_affine()
    assert out[2] is None


def test_batch_normalize():
    from ffjit.curve import _batch_normalize
    curve, G, r = ff.bn254_g1()
    pts = [k * G for k in range(2, 12)]  # Jacobian, Z != 1
    for aff, P in zip(_batch_normalize(curve, pts), pts):
        assert aff == P.to_affine()


# ---- fixed-base comb ----

def test_fixed_base_matches_double_and_add(group):
    curve, G, r = group
    T = G.precompute(window=4)
    rng = random.Random(7)
    for k in [0, 1, 2, r - 1, r, r + 5, rng.randrange(r)]:
        assert T.mul(k) == (k % r) * G
    assert T.mul(0).is_infinity


def test_fixed_base_rmul_and_repr():
    curve, G, r = ff.bn254_g1()
    T = G.precompute(window=3)
    k = 987654321987654321
    assert k * T == k * G
    assert "FixedBase" in repr(T)


def test_fixed_base_requires_order_without_glv():
    F = ff.GF(2**61 - 1)  # Mersenne prime (= 3 mod 4), no GLV setup
    curve = ff.Curve(F, 0, 7)
    # find a point by brute force over small x
    q = curve.q
    P = None
    for x in range(1, 200):
        rhs = (x * x * x + 7) % q
        y = pow(rhs, (q + 1) // 4, q)
        if y * y % q == rhs:
            P = curve.point(x, y)
            break
    assert P is not None
    with pytest.raises(ValueError):
        P.precompute()
    T = P.precompute(order=q + 1, window=2)  # any bound >= point order works
    assert T.mul(5) == 5 * P
