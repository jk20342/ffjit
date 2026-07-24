"""Tests for the Poseidon-style permutation demo (demos/poseidon.py).

The demo module lives outside the ``ffjit`` package, so we add ``demos/`` to
``sys.path`` (conftest.py already makes ``ffjit`` importable from frontend/).
"""

import random
import sys
from pathlib import Path

import pytest

DEMOS = Path(__file__).resolve().parents[3] / "demos"
if str(DEMOS) not in sys.path:
    sys.path.insert(0, str(DEMOS))

import poseidon  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_jitted_permutation_matches_reference(seed):
    rng = random.Random(seed)
    for _ in range(5):
        s = [rng.randrange(poseidon.P) for _ in range(poseidon.T)]
        got = tuple(
            int(v)
            for v in poseidon.poseidon_permutation(*(poseidon.F(x) for x in s))
        )
        assert got == poseidon.poseidon_permutation_ref(*s)


def test_permutation_matches_reference_on_edge_inputs():
    p = poseidon.P
    for s in [(0, 0, 0), (1, 0, 0), (p - 1, p - 1, p - 1), (0, 1, p - 1)]:
        got = tuple(
            int(v)
            for v in poseidon.poseidon_permutation(*(poseidon.F(x) for x in s))
        )
        assert got == poseidon.poseidon_permutation_ref(*s)


def test_hash2_deterministic():
    assert poseidon.hash2(123, 456) == poseidon.hash2(123, 456)
    assert poseidon.hash2(123, 456) == poseidon.hash2_ref(123, 456)


def test_hash2_argument_order_matters():
    assert poseidon.hash2(1, 2) != poseidon.hash2(2, 1)
