"""Differential checks for compact Montgomery intermediates."""

import pytest

import ffjit as ff

P_BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P_BLS12_381_BASE = int(
    "1a0111ea397fe69a4b1ba7b6434bacd7"
    "64774b84f38512bf6730d2a0f6b0f624"
    "1eabfffeb153ffffb9feffffffffaaab",
    16,
)
P_NIST384 = (1 << 384) - (1 << 128) - (1 << 96) + (1 << 32) - 1


@pytest.mark.parametrize(
    ("modulus", "expected_limbs"),
    [
        (P_BN254, 4),
        # This repository stores the 381-bit BLS base field in six limbs.
        (P_BLS12_381_BASE, 6),
        # An exact 384-bit modulus exercises the requested seven-limb width.
        (P_NIST384, 7),
    ],
)
def test_compact_matches_generic_scalar_batch_pow_and_inv(modulus, expected_limbs):
    field = ff.GF(modulus)
    assert ff.num_limbs(modulus) == expected_limbs

    def body(x, y):
        return x * y + x - y, x**17, x.inv()

    generic = ff.jit(limb_specialization="generic")(body)
    compact = ff.jit(limb_specialization="compact")(body)
    values = [0, 1, 2, modulus // 2, modulus - 2, modulus - 1]
    pairs = list(zip(values, reversed(values)))

    for x, y in pairs:
        expected = (
            (x * y + x - y) % modulus,
            pow(x, 17, modulus),
            0 if x == 0 else pow(x, -1, modulus),
        )
        assert tuple(map(int, generic(field(x), field(y)))) == expected
        assert tuple(map(int, compact(field(x), field(y)))) == expected

    left = ff.FieldArray(field, [x for x, _ in pairs])
    right = ff.FieldArray(field, [y for _, y in pairs])
    generic_batch = generic.map(left, right)
    compact_batch = compact.map(left, right)
    assert [out.to_ints() for out in compact_batch] == [
        out.to_ints() for out in generic_batch
    ]


def test_limb_specialization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="limb_specialization"):
        ff.jit(limb_specialization="curve-name")(lambda x: x)
