"""Diagnostics and boundary-behavior tests.

Pins down (a) the error messages users hit for untraceable Python constructs,
and (b) the reduction semantics for out-of-range integers at every public
entry point, including unreduced native buffers hitting the kernel's
``from_int`` boundary (which performs a full reduction).
"""

import random
import threading

import pytest

import ffjit as ff
from ffjit.compiler import _run
from ffjit.errors import CompileError, TraceError

P = 65537
P_BIG = 21888242871839275222246405745257275088548364400416034343698204186575808495617


# ---- tracer diagnostics ----

def test_python_branch_on_traced_value_raises():
    F = ff.GF(P)

    @ff.jit
    def bad(x):
        if x:  # not traceable
            return x
        return x + 1

    with pytest.raises(TraceError, match="control flow"):
        bad(F(3))


def test_comparison_of_traced_values_raises():
    F = ff.GF(P)

    @ff.jit
    def bad(x, y):
        return x if x == y else y

    with pytest.raises(TraceError, match="[Cc]omparison"):
        bad(F(1), F(2))


def test_float_operand_raises():
    F = ff.GF(P)

    @ff.jit
    def bad(x):
        return x * 1.5

    with pytest.raises(TraceError, match="float"):
        bad(F(3))


def test_mixing_fields_raises():
    F1, F2 = ff.GF(P), ff.GF(P_BIG)

    @ff.jit
    def bad(x, y):
        return x + y

    with pytest.raises(TraceError, match="mix"):
        bad(F1(1), F2(1))


def test_non_field_argument_raises():
    @ff.jit
    def f(x):
        return x + 1

    with pytest.raises(TypeError, match="field elements"):
        f(42)


# ---- toolchain diagnostics ----

def test_failed_tool_invocation_surfaces_stderr():
    with pytest.raises(CompileError, match="exit"):
        _run(["false"], "deliberately failing step")


def test_missing_tool_reports_command():
    with pytest.raises((CompileError, FileNotFoundError)):
        _run(["/nonexistent/tool-xyz"], "missing tool")


# ---- boundary reduction semantics ----

def test_fieldval_reduces_out_of_range_ints():
    F = ff.GF(P)
    assert int(F(-5)) == P - 5
    assert int(F(P)) == 0
    assert int(F(P + 3)) == 3
    assert int(F(10**30)) == 10**30 % P


def test_fieldarray_reduces_out_of_range_ints():
    F = ff.GF(P_BIG)
    vals = [-1, P_BIG, P_BIG + 7, 10**80]
    fa = ff.FieldArray(F, vals)
    assert fa.to_ints() == [v % P_BIG for v in vals]


def test_kernel_reduces_unreduced_native_buffers():
    """A raw limb buffer holding p+1 (not a canonical residue) must still
    produce reduced results: the kernel's from_int boundary does a full
    reduction."""
    F = ff.GF(P_BIG)

    @ff.jit
    def sq(x):
        return x * x

    nb = ff.num_limbs(P_BIG) * 8
    raw = (P_BIG + 1).to_bytes(nb, "little")
    fa = ff.FieldArray._from_raw(F, raw, 1)
    assert sq.map(fa).to_ints() == [1]  # (p+1)^2 ≡ 1 (mod p)


def test_scalar_kernel_matches_reference_on_edge_values():
    F = ff.GF(P_BIG)

    @ff.jit
    def affine(x, y):
        return x * y + x

    for a, b in [(0, 0), (P_BIG - 1, P_BIG - 1), (1, P_BIG - 1), (0, 5)]:
        assert int(affine(F(a), F(b))) == (a * b + a) % P_BIG


# ---- cache concurrency ----

def test_concurrent_compilation_of_same_kernel():
    """Two threads compiling the same (previously uncached) kernel must both
    succeed; the flock serializes the build and the .so lands atomically."""
    F = ff.GF(P)
    salt = random.randrange(1 << 48)  # ensure a fresh cache entry

    @ff.jit
    def fresh(x):
        return x * salt + 1

    results, errors = [], []

    def work():
        try:
            results.append(int(fresh(F(2))))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results == [(2 * salt + 1) % P] * 4
