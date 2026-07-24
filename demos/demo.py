"""A short tour of ffjit.

Run with:  PYTHONPATH=frontend python3 demos/demo.py
        or:  make demo
"""

import ffjit as ff

BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def banner(t):
    print("\n" + t + "\n" + "-" * len(t))


def main():
    banner("1. A word-sized field, GF(65537)")
    F = ff.GF(65537)

    @ff.jit
    def f(x, y):
        return (x * y + x).inv()

    print("generated MLIR:")
    print(f.mlir(F, F))
    print("f(3, 5) =", f(F(3), F(5)))

    banner("2. The 254-bit BN254 scalar field (no existing Python JIT handles this)")
    K = ff.GF(BN254)

    @ff.jit
    def horner(x, a, b, c):
        return (a * x + b) * x + c  # a*x^2 + b*x + c

    x, a, b, c = 7, 11, 13, 17
    got = horner(K(x), K(a), K(b), K(c))
    ref = (a * x * x + b * x + c) % BN254
    print(f"horner(7,11,13,17) = {int(got)}")
    print(f"matches reference   = {int(got) == ref}")

    banner("3. Inversion is a real field inverse")

    @ff.jit
    def inv(x):
        return x.inv()

    v = 123456789012345678901234567890
    iv = int(inv(K(v)))
    print(f"v * v^-1 mod p = {(v * iv) % BN254}  (should be 1)")

    banner("4. A tiny elliptic-curve style expression (BN254 base field arithmetic)")

    # slope of the chord through two affine points, lambda = (y2-y1)/(x2-x1)
    @ff.jit
    def chord_slope(x1, y1, x2, y2):
        return (y2 - y1) * (x2 - x1).inv()

    lam = chord_slope(K(1), K(2), K(3), K(4))
    # verify against pure Python
    ref = ((4 - 2) * pow(3 - 1, -1, BN254)) % BN254
    print(f"lambda = {int(lam)}")
    print(f"matches reference = {int(lam) == ref}")

    banner("5. Batched kernels: one compiled loop over native limb buffers")
    import random
    import time

    rng = random.Random(0)
    N = 100_000
    A = ff.FieldArray(K, [rng.randrange(BN254) for _ in range(N)])
    B = ff.FieldArray(K, [rng.randrange(BN254) for _ in range(N)])

    @ff.jit
    def mul(a, b):
        return a * b

    mul.map(A, B)  # warm-up compile
    t0 = time.perf_counter()
    out = mul.map(A, B)
    dt = time.perf_counter() - t0
    print(f"{N} BN254 multiplies in {dt*1e3:.1f} ms "
          f"({dt/N*1e9:.0f} ns per multiply)")
    print("spot check ok =", out.to_ints()[0]
          == A.to_ints()[0] * B.to_ints()[0] % BN254)

    banner("6. NTT polynomial multiplication, O(n log n)")
    n = 2048
    pa = ff.Poly(K, [rng.randrange(BN254) for _ in range(n)])
    pb = ff.Poly(K, [rng.randrange(BN254) for _ in range(n)])
    pa * pb  # warm-up: builds NTT plan, compiles butterfly kernel
    t0 = time.perf_counter()
    pc = pa * pb
    dt = time.perf_counter() - t0
    print(f"deg-{n-1} x deg-{n-1} product over BN254 Fr in {dt*1e3:.1f} ms")
    x = rng.randrange(BN254)
    print("convolution check ok =",
          int(pc(x)) == int(pa(x)) * int(pb(x)) % BN254)

    banner("7. Elliptic curves: Pippenger MSM with jitted Jacobian kernels")
    curve, G, r = ff.bn254_g1()
    m = 256
    points = [rng.randrange(1, r) * G for _ in range(m)]
    scalars = [rng.randrange(r) for _ in range(m)]
    t0 = time.perf_counter()
    S = ff.msm(points, scalars)
    dt = time.perf_counter() - t0
    print(f"MSM of {m} BN254-G1 points, 254-bit scalars: {dt*1e3:.0f} ms")
    # spot check against a direct sum of the first few terms
    check = ff.msm(points[:3], scalars[:3])
    direct = scalars[0] * points[0] + scalars[1] * points[1] \
        + scalars[2] * points[2]
    print("spot check ok =", check == direct)


if __name__ == "__main__":
    main()
