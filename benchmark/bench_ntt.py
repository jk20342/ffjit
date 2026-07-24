"""Polynomial multiplication benchmark: NTT (jitted butterflies) vs schoolbook.

Multiplies two random degree-(n-1) polynomials over the BN254 scalar field.
The NTT path is O(n log n) with compiled butterfly kernels; the reference is
O(n^2) schoolbook in CPython big ints.

Run:  PYTHONPATH=frontend python3 benchmark/bench_ntt.py
"""

import random
import time

import ffjit as ff

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
F = ff.GF(P)

SCHOOLBOOK_LIMIT = 4096  # above this the O(n^2) reference takes too long


def schoolbook(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return out


def main():
    print(f"poly mul over BN254 Fr (2-adicity {ff.two_adicity(P)}); "
          "times are best of 3")
    print(f"{'n':>6} {'ffjit NTT':>12} {'schoolbook':>12} {'speedup':>9}")
    rng = random.Random(11)
    for logn in (9, 10, 11, 12, 13):
        n = 1 << logn
        a = [rng.randrange(P) for _ in range(n)]
        b = [rng.randrange(P) for _ in range(n)]
        pa, pb = ff.Poly(F, a), ff.Poly(F, b)

        pa * pb  # warm-up: builds the NTT plan and compiles kernels
        t_ntt = min(
            _timed(lambda: pa._mul_ntt(pb, 2 * n - 1, logn + 1))
            for _ in range(3)
        )

        if n <= SCHOOLBOOK_LIMIT:
            t0 = time.perf_counter()
            ref = schoolbook(a, b, P)
            t_sb = time.perf_counter() - t0
            assert (pa * pb).coeffs == ref
            print(f"{n:>6} {t_ntt*1e3:>9.1f} ms {t_sb*1e3:>9.1f} ms "
                  f"{t_sb/t_ntt:>8.0f}x")
        else:
            print(f"{n:>6} {t_ntt*1e3:>9.1f} ms {'(skipped)':>12} {'--':>9}")


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
