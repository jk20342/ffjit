"""Poseidon-style permutation over the BN254 scalar field, JIT-compiled whole.

This demo builds the FULL Poseidon permutation (t = 3, S-box x^5, R_F = 8 full
rounds, R_P = 57 partial rounds -- the standard round numbers for this width at
~128-bit security) as a single ``@ff.jit`` function. Tracing unrolls all 65
rounds into one straight-line kernel of ~1400 field operations, which ffjit
compiles to a single native function: constants (round constants and MDS
entries) fold into the trace as Python ints.

IMPORTANT -- these are NOT the standard Poseidon-BN254 constants.
The reference Poseidon instantiation (circomlib et al.) derives its round
constants and MDS matrix with the Grain LFSR procedure from the Poseidon
paper. Here, for a self-contained educational/benchmark implementation of the
*permutation structure*, we instead derive:

  * round constants deterministically from sha256:
        rc_i = int(sha256(b"ffjit-poseidon-rc" + i.to_bytes(4, "big"))) % p
  * the MDS matrix as a 3x3 Cauchy matrix:
        M[i][j] = 1 / (x_i + y_j) mod p  with  x_i = i, y_j = t + j

A Cauchy matrix over a prime field with distinct x_i, y_j (and x_i + y_j != 0)
is always MDS, so the diffusion properties hold, but digests will NOT match
circomlib/other standard Poseidon implementations.

Run with:  PYTHONPATH=frontend python3 demos/poseidon.py
(The very first run compiles the ~1400-op kernel, which takes a few minutes;
subsequent runs load it from the on-disk cache instantly.)
"""

import hashlib
import random
import time

import ffjit as ff

# BN254 (alt_bn128) scalar field modulus.
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
F = ff.GF(P)

T = 3        # state width
R_F = 8      # full rounds (R_F/2 at the beginning, R_F/2 at the end)
R_P = 57     # partial rounds
N_ROUNDS = R_F + R_P
_HALF_F = R_F // 2


def _rc(i: int) -> int:
    """Deterministic round constant (see module docstring: non-standard)."""
    h = hashlib.sha256(b"ffjit-poseidon-rc" + i.to_bytes(4, "big")).digest()
    return int.from_bytes(h, "big") % P


# Round constants: one triple per round.
RC = [[_rc(r * T + i) for i in range(T)] for r in range(N_ROUNDS)]

# 3x3 Cauchy MDS matrix: M[i][j] = 1/(x_i + y_j), x_i = i, y_j = T + j.
MDS = [[pow(i + T + j, -1, P) for j in range(T)] for i in range(T)]


def _is_full_round(r: int) -> bool:
    return r < _HALF_F or r >= _HALF_F + R_P


# NOTE on compile time: the first call compiles all ~1400 ops into one native
# function. LLVM's wide-integer (i256/i512) codegen on a straight-line function
# this large takes ~5 minutes REGARDLESS of opt level (-O0/-O1/-O2 all measure
# within ~1.5x of each other; ffc's MLIR lowering itself takes <1 s), so we
# keep the default -O2, which gives the best runtime. The kernel is cached on
# disk (.ffjit_cache), so this cost is paid once per machine.
@ff.jit
def poseidon_permutation(s0, s1, s2):
    """The whole 65-round permutation, traced into one native kernel.

    The Python loops below run only at trace time; the constants RC/MDS are
    plain ints that fold into the straight-line trace.
    """
    state = [s0, s1, s2]
    for r in range(N_ROUNDS):
        # AddRoundConstants
        state = [state[i] + RC[r][i] for i in range(T)]
        # S-box x^5: all lanes in full rounds, lane 0 only in partial rounds
        if _is_full_round(r):
            state = [(x * x) * (x * x) * x for x in state]
        else:
            x = state[0]
            state[0] = (x * x) * (x * x) * x
        # MDS matrix-vector product
        state = [
            MDS[i][0] * state[0] + MDS[i][1] * state[1] + MDS[i][2] * state[2]
            for i in range(T)
        ]
    return state[0], state[1], state[2]


def poseidon_permutation_ref(s0: int, s1: int, s2: int):
    """Pure-Python reference for the identical permutation (plain ints % P)."""
    state = [s0 % P, s1 % P, s2 % P]
    for r in range(N_ROUNDS):
        state = [(state[i] + RC[r][i]) % P for i in range(T)]
        if _is_full_round(r):
            state = [pow(x, 5, P) for x in state]
        else:
            state[0] = pow(state[0], 5, P)
        state = [
            (MDS[i][0] * state[0] + MDS[i][1] * state[1] + MDS[i][2] * state[2]) % P
            for i in range(T)
        ]
    return state[0], state[1], state[2]


def hash2(a, b) -> int:
    """Simplified sponge: absorb (a, b) with capacity lane 0, squeeze state[0].

    This is a fixed-arity 2-to-1 compression (permute(a, b, 0), take lane 0),
    not a full padded sponge construction.
    """
    out = poseidon_permutation(F(a), F(b), F(0))
    return int(out[0])


def hash2_ref(a: int, b: int) -> int:
    """Pure-Python counterpart of :func:`hash2`."""
    return poseidon_permutation_ref(a, b, 0)[0]


def main():
    # -- 0. compile the whole permutation into one kernel (first call) --
    t0 = time.perf_counter()
    poseidon_permutation(F(0), F(0), F(0))
    dt = time.perf_counter() - t0
    n_ops = R_F * (3 + 3 * 3 + 9 + 6) + R_P * (3 + 3 + 9 + 6)
    print(f"compiled {N_ROUNDS}-round permutation (~{n_ops} field ops) "
          f"in {dt:.2f} s (cached on disk afterwards)")

    # -- 1. correctness vs the pure-Python reference --
    rng = random.Random(1234)
    for _ in range(20):
        s = [rng.randrange(P) for _ in range(T)]
        got = tuple(int(v) for v in poseidon_permutation(*(F(x) for x in s)))
        ref = poseidon_permutation_ref(*s)
        assert got == ref, f"mismatch on input {s}: {got} != {ref}"
    print("reference check ok (20 random inputs)")

    # -- 2. hash chain of length 1000: jitted vs pure Python --
    n = 1000

    t0 = time.perf_counter()
    h = 0
    for i in range(n):
        h = hash2(h, i)
    jit_time = time.perf_counter() - t0
    jit_digest = h

    t0 = time.perf_counter()
    h = 0
    for i in range(n):
        h = hash2_ref(h, i)
    py_time = time.perf_counter() - t0
    assert h == jit_digest

    print(f"hash chain of length {n}:")
    print(f"  jitted kernel : {jit_time:.3f} s "
          f"({jit_time / n * 1e6:.0f} us per hash)")
    print(f"  pure Python   : {py_time:.3f} s "
          f"({py_time / n * 1e6:.0f} us per hash)")
    print(f"  speedup       : {py_time / jit_time:.1f}x")

    # -- 3. sample digest --
    print(f"hash2(1, 2) = {hash2(1, 2)}")


if __name__ == "__main__":
    main()
