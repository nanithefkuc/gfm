# Benchmarks

Public API timings, measured 2026-09-20. Every cell is
**Core Ultra 7 258V / Core i7-12700K**, each the median of five per-run
Criterion medians. `-` means unavailable.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Arch Linux, Linux 7.2.4 | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | CachyOS, Linux 7.2.3 | 1.98.1 | 8, isolated |

| Setting | Value |
| --- | --- |
| Crate | `gfm` 1.0.0 working tree, fingerprint `2a0ca732b0de086b` |
| Dependencies | `fgf` 1.1.1, Criterion 0.8.2 |
| Build | `--all-features`, thin LTO, one codegen unit, no custom `RUSTFLAGS` |
| Threads | `RAYON_NUM_THREADS=1`; affinity verified per process |
| Field | `Gf8B` — GF(256) under the AES polynomial `0x11B` |
| Resolved backend | `v3_gfni_crypto` on both hosts |
| Aggregation | Median of five per-run Criterion medians, three significant figures |

The measured tree is a working snapshot rather than a release commit, and
`just test` passed on both hosts before timing.

## Hybrid versus dense solve

1040 equations over 1000 unknowns, 1024-byte symbols, LT-degree binary
coefficients, zero right-hand side.

| Operation | Time (ms) |
| --- | --- |
| `Hybrid::solve_into` | 0.385/0.389 |
| `Ple::decompose` + `Ple::solve_into` | 8.39/6.77 |

Hybrid reuses populated state and output buffers. The dense arm includes
matrix cloning and a fresh decomposition, reusing output and scratch.

## Dense decomposition

Square matrices, including input cloning and result drop, with reused
scratch. These fixtures are low-rank rather than full-rank workloads.

| Order | Rank | `Ple::decompose` (µs) |
| --- | --- | --- |
| 128 | 2 | 13.9/13.8 |
| 256 | 1 | 53.9/48.9 |
| 512 | 1 | 224/185 |
| 1024 | 1 | 1010/994 |

## Small-matrix solve

Full-rank square coefficients, 1024 right-hand-side columns. Each call
includes preparation — `SmallMatrix::from_matrix` or `Ple::decompose` — and
the solve; outputs are preallocated and PLE scratch is reused.

| Order | `SmallMatrix` (µs) | `Ple` (µs) |
| --- | --- | --- |
| 4 | 0.256/0.245 | 0.416/0.378 |
| 8 | 1.12/1.09 | 1.54/1.38 |
| 16 | 5.01/4.85 | 6.39/5.97 |
| 32 | 20.2/19.6 | 24.8/22.5 |
| 64 | 85.3/77.5 | 117/104 |

## Hybrid scale

Synthetic LT-degree rows plus `ceil(sqrt(columns)) + 8` overhead rows,
64-byte symbols, zero right-hand side, output allocation included. HDPC rows
repeat one coefficient vector. These are solver workloads, not complete
RFC 6330 codec measurements.

| Rows supplied | Columns | HDPC rows | `Hybrid::solve_into` (ms) |
| --- | --- | --- | --- |
| LT only | 1000 | 0 | 0.203/0.187 |
| LT only | 5000 | 0 | 1.36/1.24 |
| LT only | 10000 | 0 | 3.31/3.06 |
| LT only | 20000 | 0 | 7.71/7.00 |
| Eager HDPC | 500 | 33 | 5.88/5.93 |
| Deferred HDPC | 500 | 33 | 0.324/0.299 |
| Eager HDPC | 1000 | 58 | 35.9/32.4 |
| Deferred HDPC | 1000 | 58 | 0.857/0.810 |
| Eager HDPC | 2000 | 108 | 251/226 |
| Deferred HDPC | 2000 | 108 | 2.91/2.68 |
| Eager HDPC | 4000 | 208 | 1890/1770 |
| Deferred HDPC | 4000 | 208 | 11.3/9.54 |
| Deferred HDPC | 56403 | 108 | 119/101 |
| Deferred HDPC | 20000 | 1008 | - |
| Deferred HDPC | 56403 | 2828 | - |

The harness rejects a duplicate benchmark identifier after the first
thirteen cases, so the two largest deferred geometries are unreachable.

## Rank against computer-algebra libraries

Square rank over deterministic dense inputs. Every arm builds its own matrix
from one immutable input inside the timed region, so import and allocation
costs are included for all libraries. The `gfm` arms are
`BitMatrix::from_rows` with `bits::Ple::decompose` for GF(2), and
`Matrix::from_rows` with `Ple::decompose` for GF(256).

| Field | Order | `gfm` (µs) | M4RI (µs) | FFLAS-FFPACK (µs) | M4RIE (µs) | FLINT (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| GF(2) | 128 | 52.0/61.5 | 67.4/86.1 | 975/870 | - | - |
| GF(2) | 256 | 337/317 | 344/323 | 4950/4540 | - | - |
| GF(2) | 512 | 1300/1170 | 1530/1340 | 31900/27600 | - | - |
| GF(256) | 128 | 148/129 | - | - | 476/490 | 22400/20500 |
| GF(256) | 256 | 637/540 | - | - | 1320/1230 | 118000/108000 |
| GF(256) | 512 | 3060/2440 | - | - | 5410/4510 | 616000/556000 |

| Library | Version | Entry point | Field | License |
| --- | --- | --- | --- | --- |
| M4RI | 20260122 | `mzd_echelonize` | GF(2) | GPL |
| FFLAS-FFPACK | 2.5.0 | `FFPACK::Rank` over `Givaro::Modular` | GF(2) | LGPL |
| M4RIE | 20250128 | `mzed_ple` | GF(256), `0x11B` | GPL |
| FLINT | 3.6.0 | `fq_nmod_mat_lu` | GF(256), `0x11B` | LGPL |

All four are copyleft, so the comparison harness and its adapters are
neither distributed with this crate nor part of its build; only the measured
numbers appear here. Reproducing the comparison means rebuilding an
equivalent harness against these versions. FLINT and M4RIE were configured
with the `0x11B` reduction polynomial, matching `Gf8B`, so element encodings
agree byte for byte. Each arm's rank was checked against `gfm` before
timing, and all arms ran single-threaded.

## Unavailable coverage

| Operation | Scope | Time |
| --- | --- | ---: |
| `bits::Ple::decompose` | Solve, kernel, inverse over GF(2) | - |
| `Ple::decompose` | `Gf16`, `Gf32`, `Gf64` | - |
| `Echelon` | Incremental row insertion | - |

The benchmark targets carry no public-path cases for these operations.

## Reproduction

Set `FEC_GOLDEN_CORE` to the host's pinned core and run from the crate root:

```sh
export FEC_GOLDEN_CORE=<cpu> RAYON_NUM_THREADS=1
for run in 1 2 3 4 5; do
    just _bench-run hybrid --noplot --save-baseline "public-$run"
    just _bench-run tuning --noplot --save-baseline "public-$run" \
        "'^(small_matrix/(small|ple)/(4|8|16|32|64)|dense_newton_john/blocked/(128|256|512|1024))$'"
    just _bench-run rfc_scale --noplot --save-baseline "public-$run"
done
```

| Detail | Value |
| --- | --- |
| Source of each number | `median.point_estimate` in `target/criterion/**/public-*/estimates.json`, in nanoseconds |
| Samples | 10 per case per run |
| Warm-up / measurement | hybrid 3 s/5 s, dense 3 s/3 s, small 0.5 s/1 s, scale 1 ms/5 s; Criterion extends slow cases |
| Retained after failure | `rfc_scale` estimates saved before its duplicate-identifier error |

Paired cells are independent per-host measurements, not paired ratios, and
no cross-host speedup or across-run confidence interval is reported.
