"""Performance regression harness.

Runs a fixed set of small, stable benchmarks and compares the timings
against a locally recorded baseline (JSON). Baselines are machine-specific,
so the file lives outside version control (.ffjit_perf.json by default):

    PYTHONPATH=frontend python3 benchmark/perf.py --save     # record baseline
    PYTHONPATH=frontend python3 benchmark/perf.py            # check against it

A benchmark fails the check when it is more than --tolerance times slower
than the baseline (default 1.6x, generous enough for machine noise but
tight enough to catch algorithmic regressions). Exit code 1 on regression.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import ffjit as ff  # noqa: E402
from ffjit.curve import _msm_ref  # noqa: E402
from ffjit.ntt import NegacyclicPlan, get_plan  # noqa: E402

P_BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P_M61 = 2**61 - 1


def _timeit(fn, *, repeat: int = 5, number: int = 1) -> float:
    """Median wall-clock seconds of `number` calls, over `repeat` samples."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - t0) / number)
    return statistics.median(samples)


def bench_scalar_call() -> float:
    """Per-call latency of a compiled scalar kernel (FFI-dominated)."""
    F = ff.GF(P_BN254_R)

    @ff.jit
    def f(x, y):
        return x * y + x

    a, b = F(12345), F(67890)
    f(a, b)  # compile
    n = 2000
    return _timeit(lambda: [f(a, b) for _ in range(n)]) / n


def bench_batch_mul() -> float:
    """Throughput of a batched mul over 100k Mersenne61 elements."""
    F = ff.GF(P_M61)

    @ff.jit
    def mul(x, y):
        return x * y

    n = 100_000
    A = ff.FieldArray(F, [(3 * i + 1) % P_M61 for i in range(n)])
    B = ff.FieldArray(F, [(7 * i + 2) % P_M61 for i in range(n)])
    mul.map(A, B)  # compile
    return _timeit(lambda: mul.map(A, B))


def bench_ntt() -> float:
    """Forward NTT of size 4096 over the BN254 scalar field."""
    F = ff.GF(P_BN254_R)
    fa = ff.FieldArray(F, [i % P_BN254_R for i in range(4096)])
    ff.ntt(fa)  # compile + plan
    return _timeit(lambda: ff.ntt(fa))


def bench_ntt_reference() -> float:
    """Original Python-staged NTT, retained as the native-path oracle."""
    F = ff.GF(P_BN254_R)
    fa = ff.FieldArray(F, [i % P_BN254_R for i in range(4096)])
    plan = get_plan(F, 12)
    plan._transform_ref(fa, plan.tw_fwd)
    return _timeit(lambda: plan._transform_ref(fa, plan.tw_fwd))


def bench_fused_poly_mul() -> float:
    """One-call cyclic convolution over 4096 BN254 coefficients."""
    F = ff.GF(P_BN254_R)
    plan = get_plan(F, 12)
    a = ff.FieldArray(F, range(4096))
    b = ff.FieldArray(F, range(1, 4097))
    plan.mul(a, b)
    return _timeit(lambda: plan.mul(a, b), repeat=3)


def bench_fused_negacyclic_mul() -> float:
    """One-call negacyclic convolution over 1024 BN254 coefficients."""
    F = ff.GF(P_BN254_R)
    plan = NegacyclicPlan(F, 10)
    a = ff.FieldArray(F, range(1024))
    b = ff.FieldArray(F, range(1, 1025))
    plan.mul(a, b)
    return _timeit(lambda: plan.mul(a, b), repeat=3)


def bench_msm() -> float:
    """Default Pippenger MSM, 128 points on BN254 G1."""
    curve, G, r = ff.bn254_g1()
    pts = [k * G for k in range(2, 130)]
    ks = [pow(k, 99, r) for k in range(2, 130)]
    ff.msm(pts[:4], ks[:4])  # compile
    return _timeit(lambda: ff.msm(pts, ks), repeat=3)


def bench_msm_native() -> float:
    """Schedule-native Pippenger MSM on the same 128-point input."""
    curve, G, r = ff.bn254_g1()
    pts = [k * G for k in range(2, 130)]
    ks = [pow(k, 99, r) for k in range(2, 130)]
    previous = os.environ.get("FFJIT_NATIVE_MSM")
    os.environ["FFJIT_NATIVE_MSM"] = "strict"
    try:
        ff.msm(pts[:4], ks[:4])
        return _timeit(lambda: ff.msm(pts, ks), repeat=3)
    finally:
        if previous is None:
            os.environ.pop("FFJIT_NATIVE_MSM", None)
        else:
            os.environ["FFJIT_NATIVE_MSM"] = previous


def bench_msm_reference() -> float:
    """Original Python Pippenger scheduler on the same 128-point input."""
    curve, G, r = ff.bn254_g1()
    pts = [k * G for k in range(2, 130)]
    ks = [pow(k, 99, r) for k in range(2, 130)]
    _msm_ref(pts[:4], ks[:4])
    return _timeit(lambda: _msm_ref(pts, ks), repeat=3)


def bench_inversion(strategy: str) -> float:
    """Batch inversion throughput for one compiler lowering strategy."""
    F = ff.GF(P_BN254_R)

    @ff.jit(inv=strategy)
    def invert(x):
        return x.inv()

    values = ff.FieldArray(F, range(1, 257))
    invert.map(values)
    return _timeit(lambda: invert.map(values), repeat=3)


def bench_fixed_base() -> float:
    """Fixed-base comb scalar multiplication on BN254 G1."""
    curve, G, r = ff.bn254_g1()
    T = G.precompute(window=8)
    k = pow(31337, 1234567, r)
    n = 20
    return _timeit(lambda: [T.mul(k) for _ in range(n)], repeat=3) / n


BENCHES = {
    "scalar_call_s": bench_scalar_call,
    "batch_mul_100k_s": bench_batch_mul,
    "ntt_4096_s": bench_ntt,
    "ntt_ref_4096_s": bench_ntt_reference,
    "poly_fused_4096_s": bench_fused_poly_mul,
    "negacyclic_fused_1024_s": bench_fused_negacyclic_mul,
    "msm_128_s": bench_msm,
    "msm_native_128_s": bench_msm_native,
    "msm_ref_128_s": bench_msm_reference,
    "fixed_base_mul_s": bench_fixed_base,
    "inv_fermat_256_s": lambda: bench_inversion("fermat"),
    "inv_runtime_256_s": lambda: bench_inversion("runtime"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=".ffjit_perf.json",
                    help="baseline JSON path (default: .ffjit_perf.json)")
    ap.add_argument("--save", action="store_true",
                    help="record current timings as the new baseline")
    ap.add_argument("--tolerance", type=float, default=1.6,
                    help="max allowed slowdown vs baseline (default 1.6)")
    args = ap.parse_args()

    print("running perf benchmarks...")
    current = {}
    for name, fn in BENCHES.items():
        current[name] = fn()
        print(f"  {name:22s} {current[name] * 1e3:10.3f} ms")

    path = Path(args.baseline)
    if args.save or not path.exists():
        path.write_text(json.dumps(current, indent=2) + "\n")
        verb = "saved" if args.save else "no baseline found; saved"
        print(f"{verb} baseline to {path}")
        return 0

    baseline = json.loads(path.read_text())
    failed = []
    print(f"\nvs baseline {path} (tolerance {args.tolerance}x):")
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
            print(f"  {name:22s} (new benchmark, no baseline)")
            continue
        ratio = cur / base
        flag = ""
        if ratio > args.tolerance:
            flag = "  <-- REGRESSION"
            failed.append(name)
        elif ratio < 1 / args.tolerance:
            flag = "  (improved -- consider --save)"
        print(f"  {name:22s} {ratio:6.2f}x{flag}")

    if failed:
        print(f"\nFAIL: {len(failed)} regression(s): {', '.join(failed)}")
        return 1
    print("\nall benchmarks within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
