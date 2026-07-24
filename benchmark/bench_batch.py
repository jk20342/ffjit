"""Batched-kernel throughput benchmark.

Measures elementwise field kernels over pre-marshalled ``FieldArray`` buffers
(the realistic deployment: data lives in native limb buffers, Python only
orchestrates). Compares against:

  * pure CPython big-int arithmetic (per element)
  * galois (numpy-based), where field construction is feasible

Kernels:
  * mul:    r = a * b                       (1 field mul)
  * horner: degree-16 polynomial in a       (16 mul + 16 add)

Run:  PYTHONPATH=frontend python3 benchmark/bench_batch.py
"""

import os
import random
import signal
import tempfile
import time

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import ffjit as ff

P_BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P_M61 = (1 << 61) - 1  # Mersenne prime small enough for galois

N = 100_000
GALOIS_TIMEOUT_S = 30

COEFFS_SEED = 20260724
DEGREE = 16


def make_kernels():
    rng = random.Random(COEFFS_SEED)
    coeffs = [rng.randrange(1, 1 << 60) for _ in range(DEGREE + 1)]

    @ff.jit
    def mul(a, b):
        return a * b

    def make_horner(field):
        cs = [field(c) for c in coeffs]

        @ff.jit
        def horner(a):
            acc = cs[0]
            for c in cs[1:]:
                acc = acc * a + c
            return acc

        return horner

    return mul, make_horner, coeffs


def bench(fn, *args, trials=5):
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def python_mul(ai, bi, p):
    return [(x * y) % p for x, y in zip(ai, bi)]


def python_horner(ai, coeffs, p):
    out = []
    for x in ai:
        acc = coeffs[0]
        for c in coeffs[1:]:
            acc = (acc * x + c) % p
        out.append(acc)
    return out


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def try_galois(p, ai, bi, coeffs):
    """Return (t_mul, t_horner) or None if galois unusable for this prime."""
    try:
        import galois
        import numpy as np
    except Exception:
        return None
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(GALOIS_TIMEOUT_S)
    try:
        GFg = galois.GF(p, verify=False)
        A = GFg(ai)
        B = GFg(bi)
        signal.alarm(0)
    except (_Timeout, Exception):
        signal.alarm(0)
        return None

    t_mul = bench(lambda: A * B)

    cs = [GFg(c % p) for c in coeffs]

    def gh():
        r = cs[0]
        for c in cs[1:]:
            r = r * A + c
        return r

    t_horner = bench(gh)
    return t_mul, t_horner


def run_field(name, p):
    print(f"\n=== {name} (p ~ 2^{p.bit_length()}), N = {N} ===")
    F = ff.GF(p)
    rng = random.Random(1)
    ai = [rng.randrange(p) for _ in range(N)]
    bi = [rng.randrange(p) for _ in range(N)]

    mul, make_horner, coeffs = make_kernels()
    horner = make_horner(F)

    t0 = time.perf_counter()
    A = ff.FieldArray(F, ai)
    B = ff.FieldArray(F, bi)
    t_marshal = time.perf_counter() - t0

    # warm-up (compiles kernels)
    t0 = time.perf_counter()
    mul.map(A, B)
    horner.map(A)
    t_compile = time.perf_counter() - t0

    t_mul = bench(lambda: mul.map(A, B))
    t_horner = bench(lambda: horner.map(A))

    # correctness spot-check
    assert mul.map(A, B).to_ints()[:100] == python_mul(ai[:100], bi[:100], p)
    assert horner.map(A).to_ints()[:50] == python_horner(ai[:50], coeffs, p)

    tp_mul = bench(lambda: python_mul(ai, bi, p), trials=3)
    tp_horner = bench(lambda: python_horner(ai, coeffs, p), trials=1)

    print(f"  marshal once into FieldArray : {t_marshal*1e3:9.2f} ms")
    print(f"  compile (first call, cached) : {t_compile*1e3:9.2f} ms")
    print(f"  {'kernel':<10} {'ffjit':>12} {'python':>12} {'speedup':>9}")
    print(f"  {'mul':<10} {t_mul*1e3:>9.2f} ms {tp_mul*1e3:>9.2f} ms "
          f"{tp_mul/t_mul:>8.1f}x")
    print(f"  {'horner16':<10} {t_horner*1e3:>9.2f} ms {tp_horner*1e3:>9.2f} ms "
          f"{tp_horner/t_horner:>8.1f}x")
    print(f"  ffjit ns/elem: mul {t_mul/N*1e9:.0f}, horner16 {t_horner/N*1e9:.0f}")

    g = try_galois(p, ai, bi, coeffs)
    if g is None:
        print(f"  galois: field construction not feasible within "
              f"{GALOIS_TIMEOUT_S}s (or unavailable) for this prime")
    else:
        gm, gh = g
        print(f"  galois: mul {gm*1e3:.2f} ms ({gm/t_mul:.1f}x vs ffjit), "
              f"horner16 {gh*1e3:.2f} ms ({gh/t_horner:.1f}x vs ffjit)")


def main():
    print("ffjit batched-kernel benchmark (best of several trials)")
    run_field("Mersenne61", P_M61)
    run_field("BN254 scalar field", P_BN254)


if __name__ == "__main__":
    main()
