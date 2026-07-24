"""Multi-scalar multiplication benchmark over BN254 G1.

Computes sum_i k_i * P_i for N random points and full-width scalars.
Compared implementations:

  * ffjit Pippenger -- bucket method, additions batched through the
    JIT-compiled Jacobian add kernel
  * ffjit per-point -- jitted double-and-add per point, then sum
    (measures what compiled kernels buy WITHOUT the MSM algorithm)
  * pure Python     -- affine double-and-add per point with pow(x,-1,q)
    inversions (the typical naive prototype code)

Run:  PYTHONPATH=frontend python3 benchmark/bench_msm.py
"""

import random
import time

import ffjit as ff


def python_affine_msm(curve, pts_aff, ks):
    q, a = curve.q, curve.a

    def add(P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        (x1, y1), (x2, y2) = P, Q
        if x1 == x2 and (y1 + y2) % q == 0:
            return None
        if P == Q:
            lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, q) % q
        else:
            lam = (y2 - y1) * pow(x2 - x1, -1, q) % q
        x3 = (lam * lam - x1 - x2) % q
        return (x3, (lam * (x1 - x3) - y1) % q)

    acc = None
    for P, k in zip(pts_aff, ks):
        s = None
        for bit in bin(k)[2:]:
            s = add(s, s)
            if bit == "1":
                s = add(s, P)
        acc = add(acc, s)
    return acc


def main():
    curve, G, r = ff.bn254_g1()
    rng = random.Random(2026)

    print("MSM over BN254 G1, full 254-bit scalars")
    print(f"{'N':>6} {'Pippenger':>12} {'per-point jit':>14} "
          f"{'pure Python':>12} {'speedup':>9}")

    for n in (128, 512, 2048):
        pts = [rng.randrange(1, r) * G for _ in range(n)]
        ks = [rng.randrange(r) for _ in range(n)]

        ff.msm(pts[:4], ks[:4])  # warm-up (kernels already compiled anyway)

        t0 = time.perf_counter()
        got = ff.msm(pts, ks)
        t_pip = time.perf_counter() - t0

        if n <= 512:
            t0 = time.perf_counter()
            acc = curve.infinity()
            for P, k in zip(pts, ks):
                acc = acc + k * P
            t_pp = time.perf_counter() - t0
            assert acc == got

            pts_aff = [P.to_affine() for P in pts]
            t0 = time.perf_counter()
            ref = python_affine_msm(curve, pts_aff, ks)
            t_py = time.perf_counter() - t0
            assert ref == got.to_affine()

            print(f"{n:>6} {t_pip*1e3:>9.0f} ms {t_pp*1e3:>11.0f} ms "
                  f"{t_py*1e3:>9.0f} ms {t_py/t_pip:>8.1f}x")
        else:
            print(f"{n:>6} {t_pip*1e3:>9.0f} ms {'--':>14} {'--':>12} {'--':>9}")

    print("\nper-point jit = compiled kernels but naive algorithm;")
    print("Pippenger's gain over it is pure algorithm + batching.")


if __name__ == "__main__":
    main()
