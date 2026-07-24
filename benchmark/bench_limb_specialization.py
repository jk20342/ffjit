"""Compare generic and compact Montgomery intermediate widths.

Run:
  PYTHONPATH=frontend python3 benchmark/bench_limb_specialization.py
"""

import ctypes
import random
import time

import ffjit as ff
from ffjit.compiler import compile_raw_module
from ffjit.ntt import NTTPlan
from ffjit.nttgen import generate_ntt_module

P_BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def _measure(functions, trials):
    samples = {name: [] for name in functions}
    for trial in range(trials):
        order = list(functions)
        if trial % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter_ns()
            functions[name]()
            samples[name].append(time.perf_counter_ns() - start)
    return samples


def _report(label, samples, work_items):
    print(label)
    for name in ("generic", "compact"):
        best = min(samples[name])
        median = sorted(samples[name])[len(samples[name]) // 2]
        print(
            f"  {name:7s}: best {best / 1e6:9.3f} ms, "
            f"median {median / 1e6:9.3f} ms, "
            f"best {best / work_items:8.2f} ns/item"
        )
    ratio = min(samples["generic"]) / min(samples["compact"])
    print(f"  compact speed ratio: {ratio:.4f}x")


def benchmark_batch_mul():
    field = ff.GF(P_BN254)
    rng = random.Random(20260724)
    count = 200_000
    left = ff.FieldArray(field, [rng.randrange(P_BN254) for _ in range(count)])
    right = ff.FieldArray(field, [rng.randrange(P_BN254) for _ in range(count)])

    def body(a, b):
        return a * b

    kernels = {
        mode: ff.jit(limb_specialization=mode)(body) for mode in ("generic", "compact")
    }
    outputs = {mode: kernels[mode].map(left, right) for mode in kernels}
    assert outputs["generic"].as_bytes() == outputs["compact"].as_bytes()
    samples = _measure(
        {mode: lambda k=kernel: k.map(left, right) for mode, kernel in kernels.items()},
        trials=11,
    )
    _report("BN254 batch multiply (200000 elements)", samples, count)


def benchmark_native_ntt():
    field = ff.GF(P_BN254)
    logn = 12
    count = 1 << logn
    plan = NTTPlan(field, logn)
    module = generate_ntt_module(
        "ff_bench_limb_specialization_ntt",
        P_BN254,
        logn,
        inverse=False,
    )
    kernels = {
        mode: compile_raw_module(module, limb_specialization=mode)
        for mode in ("generic", "compact")
    }
    rng = random.Random(20260725)
    source = ff.FieldArray(field, [rng.randrange(P_BN254) for _ in range(count)])
    outputs = {
        mode: ctypes.create_string_buffer(count * field.num_limbs * 8)
        for mode in kernels
    }

    def run(mode):
        kernels[mode](
            [
                ctypes.addressof(outputs[mode]),
                source.buffer_address(),
                plan._native_tw_fwd.buffer_address(),
            ]
        )

    run("generic")
    run("compact")
    assert outputs["generic"].raw == outputs["compact"].raw
    samples = _measure(
        {mode: lambda m=mode: run(m) for mode in kernels},
        trials=11,
    )
    _report("BN254 native forward NTT (4096 points)", samples, count)


def main():
    print("limb specialization benchmark (alternating, best and median of 11)")
    benchmark_batch_mul()
    benchmark_native_ntt()


if __name__ == "__main__":
    main()
