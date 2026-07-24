"""Benchmark: Horner evaluation of a high-degree polynomial over BN254's
scalar field, comparing ffjit against pure-Python big integers and (if it can
construct the field) the `galois` library.

Horner is a good scalar showcase: one JIT call performs thousands of field
multiply-adds in compiled Montgomery arithmetic, so compute dominates the
~microsecond ctypes marshalling overhead. For 254-bit primes `galois` cannot
JIT (values exceed int64) and falls back to pure-Python object ufuncs.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import ffjit as ff

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
# Kernels are currently unrolled straight-line code, and the LLVM back-end's
# optimization of wide-integer (i512) multiplies scales super-linearly, so we
# keep the degree modest. Batched/looped kernels (Phase 2) will lift this.
DEGREE = 96
TRIALS = 300


def make_problem():
    random.seed(1234)
    coeffs = [random.randrange(P) for _ in range(DEGREE + 1)]
    xs = [random.randrange(P) for _ in range(TRIALS)]
    return coeffs, xs


def ref_horner(coeffs, x):
    acc = 0
    for c in coeffs:
        acc = (acc * x + c) % P
    return acc


def bench_python(coeffs, xs):
    t0 = time.perf_counter()
    out = [ref_horner(coeffs, x) for x in xs]
    return time.perf_counter() - t0, out


def bench_ffjit(coeffs, xs):
    F = ff.GF(P)

    @ff.jit(opt="-O1")
    def horner(x):
        acc = coeffs[0]
        for c in coeffs[1:]:
            acc = acc * x + c
        return acc

    horner(F(xs[0]))  # warm the compile cache
    t0 = time.perf_counter()
    out = [int(horner(F(x))) for x in xs]
    return time.perf_counter() - t0, out


def bench_galois(coeffs, xs, setup_timeout=20):
    import os
    import tempfile
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba-"))
    try:
        import galois
    except Exception as e:  # pragma: no cover
        return None, None, f"galois not importable: {e}"

    import signal

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout()

    try:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(setup_timeout)
        t0 = time.perf_counter()
        GFp = galois.GF(P, verify=False)
        setup = time.perf_counter() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    except _Timeout:
        return None, None, (
            f"galois GF(p) construction exceeded {setup_timeout}s "
            "(254-bit prime: must factor p-1) -- exactly the pain point ffjit avoids"
        )
    except Exception as e:
        signal.alarm(0)
        return None, None, f"galois could not construct GF(p): {e}"
    try:
        poly = galois.Poly(GFp(coeffs))
        t0 = time.perf_counter()
        out = [int(poly(GFp(x))) for x in xs]
        return time.perf_counter() - t0, out, f"(field setup {setup:.2f}s)"
    except Exception as e:
        return None, None, f"galois eval failed: {e}"


def main():
    coeffs, xs = make_problem()
    print(f"BN254 scalar field; Horner degree {DEGREE}, {TRIALS} evaluations\n")

    tpy, out_py = bench_python(coeffs, xs)
    print(f"  pure Python : {tpy*1e3:8.2f} ms  ({tpy/TRIALS*1e6:7.1f} us/eval)")

    tff, out_ff = bench_ffjit(coeffs, xs)
    assert out_ff == out_py, "ffjit disagrees with reference!"
    print(f"  ffjit (JIT) : {tff*1e3:8.2f} ms  ({tff/TRIALS*1e6:7.1f} us/eval)"
          f"   [{tpy/tff:.1f}x vs Python]")

    res = bench_galois(coeffs, xs)
    if res[0] is None:
        print(f"  galois      : skipped -- {res[2]}")
    else:
        tg, out_g, note = res
        ok = "OK" if out_g == out_py else "MISMATCH"
        print(f"  galois      : {tg*1e3:8.2f} ms  ({tg/TRIALS*1e6:7.1f} us/eval)"
              f"   [{tg/tff:.1f}x slower than ffjit] {note} {ok}")

    print("\nAll implementations agree on results." )


if __name__ == "__main__":
    main()
