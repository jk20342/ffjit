"""Cross-check the three inversion implementations against each other:

  * the C ABI runtime (binary extended Euclid),
  * the JIT (pure-IR Fermat ladder),
  * pure-Python ``pow(a, -1, p)``.

Skipped automatically if the shared runtime library has not been built.
"""

import pytest

import ffjit as ff

try:
    from ffjit.runtime import get_runtime
    RT = get_runtime()
except Exception as e:  # pragma: no cover
    RT = None
    _reason = str(e)

pytestmark = pytest.mark.skipif(RT is None, reason="libff_rt.so not built")

PRIMES = [
    65537,
    (1 << 61) - 1,
    (1 << 127) - 1,
    21888242871839275222246405745257275088548364400416034343698204186575808495617,
]


def test_abi_version():
    assert RT.abi_version == 1


@pytest.mark.parametrize("p", PRIMES)
def test_three_way_inverse_agreement(p):
    F = ff.GF(p)

    @ff.jit
    def jinv(x):
        return x.inv()

    for a in [1, 2, 3, 7, p - 1, p // 3, 999999999 % p]:
        if a == 0:
            continue
        py = pow(a, -1, p)
        rt = RT.inv(a, p)
        jt = int(jinv(F(a)))
        assert py == rt == jt, f"mismatch for a={a} mod {p}: py={py} rt={rt} jit={jt}"

    assert RT.inv(0, p) == 0
