# Security policy

## Not constant-time -- do not use with secrets

`ffjit` is a research and prototyping tool. The generated code is **not
constant-time**: Montgomery reduction uses conditional subtractions, scalar
multiplication branches on scalar bits, and Python orchestrates control flow
based on values. Timing side channels can leak secret inputs.

Do **not** use `ffjit` to process private keys, secret scalars, witnesses, or
any other confidential material. It is intended for algorithm prototyping,
testing, and benchmarking with public or synthetic data -- the same posture as
the `galois` library.

## Code execution model

`ffjit` compiles and `dlopen`s shared objects at runtime (via `ffc` and
`clang`), and caches them in `.ffjit_cache/` (override with `FFJIT_CACHE`).
Only run it in environments where executing locally generated native code is
acceptable, and do not point `FFJIT_CACHE` at directories writable by
untrusted users.

## Reporting

Open a GitHub issue for correctness bugs. There is no bug-bounty program.
