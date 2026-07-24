"""Cross-check the three inversion implementations against each other:

  * the C ABI runtime (binary extended Euclid),
  * the JIT pure-IR Fermat ladder and runtime-call lowering,
  * pure-Python ``pow(a, -1, p)``.

Skipped automatically if the shared runtime library has not been built.
"""

import pytest

import ffjit as ff
from ffjit.compiler import compile_module
from ffjit.mlirgen import GeneratedModule

try:
    from ffjit.runtime import RUNTIME_ABI_VERSION, get_runtime
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
    assert RT.abi_version == RUNTIME_ABI_VERSION


def test_runtime_required_module_links_and_loads():
    text = """\
func.func private @ff_rt_abi_version() -> i32
func.func @runtime_probe() -> i64 {
  %v = func.call @ff_rt_abi_version() : () -> i32
  %w = arith.extui %v : i32 to i64
  return %w : i64
}
"""
    mod = GeneratedModule(
        text, "runtime_probe", [], [64], requires_runtime=True
    )

    assert compile_module(mod)([]) == RUNTIME_ABI_VERSION


@pytest.mark.parametrize("p", PRIMES)
def test_three_way_inverse_agreement(p):
    F = ff.GF(p)

    @ff.jit
    def fermat_inv(x):
        return x.inv()

    @ff.jit(inv="runtime")
    def runtime_inv(x):
        return x.inv()

    @ff.jit(inv="runtime", montgomery=False)
    def runtime_inv_naive(x):
        return x.inv()

    for a in [1, 2, 3, 7, p - 1, p // 3, 999999999 % p]:
        if a == 0:
            continue
        py = pow(a, -1, p)
        rt = RT.inv(a, p)
        fermat = int(fermat_inv(F(a)))
        runtime_mont = int(runtime_inv(F(a)))
        runtime_naive = int(runtime_inv_naive(F(a)))
        assert py == rt == fermat == runtime_mont == runtime_naive

    assert RT.inv(0, p) == 0
