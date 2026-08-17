# Benchmarks

Numbers that justify dispatch and layout decisions live here, and only here.
Doc comments state the decision and the mechanism, then point at this file.

Measurement hygiene, inherited from `fgf`: interleave base/new, take the
maximum of at least three runs per key, keep an unchanged 1.00x control, and
treat 16–128 byte rows as noise. Benchmarks are not CI correctness checks.

## Row pitch padding cost

Not a timed benchmark — a geometry measurement, identical on every host,
toolchain, and backend. `Matrix<F>` rounds each row up to a 32-byte pitch
(`dense::layout::ALIGN`), so a row of `w` live bytes costs `pitch(w) - w`
padding bytes, with `pitch(w) = ceil(w / 32) * 32`. The overhead ratio is
bounded by `1 + 31/w`, and is an integer-ratio step function: it spikes just
above each multiple of 32 and decays as `1/w`.

Across the row widths the benchmark suite exercises:

| Live row bytes | Pitch | Overhead |
| ---: | ---: | ---: |
| 32 | 32 | 1.0000 |
| 64 | 64 | 1.0000 |
| 128 | 128 | 1.0000 |
| 256 | 256 | 1.0000 |
| 512 | 512 | 1.0000 |
| 560 | 576 | 1.0286 |
| 1000 | 1024 | 1.0240 |
| 1024 | 1024 | 1.0000 |
| 1100 | 1120 | 1.0182 |
| 1500 | 1504 | 1.0027 |
| 2048 | 2048 | 1.0000 |
| 3000 | 3008 | 1.0027 |
| 4096 | 4096 | 1.0000 |
| 8192 | 8192 | 1.0000 |
| 16384 | 16384 | 1.0000 |
| 65536 | 65536 | 1.0000 |

Worst case over the suite above 512 bytes: **1.0286** (560 → 576), under the
5% bound the layout budget allows. The absolute worst case is narrower than
the suite bound suggests: the overhead stays under 5% only for live rows of
**621 bytes or more** (at 513 bytes it peaks at 1.0604). Rows that small are
the panel/blocking regime, where the padding is amortized out of the cost
model — the property is asserted for the suite shapes in
`tests/dense.rs::padding_cost_is_bounded`.

The GF(2) domain rounds rows to a 64-byte pitch (`bits::ALIGN`); the same
analysis applies with 63 in place of 31, i.e. `1 + 63/w`, 5% from 1.3 KiB.

## Cauchy inverse: closed form vs `Ple`

`benches/cauchy_inverse.rs`, Intel Core Ultra 7 258V, rustc 1.93, `cargo bench
--bench cauchy_inverse`. Both paths produce a full `k × k` GF(2^8) inverse from
scratch: `closed_form` is [`Cauchy::inverse_into`] (rational-Lagrange, `O(k²)`
scalar field ops), `ple` is `Ple::decompose` + `inverse_into` (`O(k³)`, but the
row updates run `fgf`'s SIMD byte kernels).

| k | closed form | `Ple` | closed / `Ple` |
| ---: | ---: | ---: | ---: |
| 16 | 4.89 µs | 4.30 µs | 1.14 |
| 32 | 21.19 µs | 16.98 µs | 1.25 |
| 48 | 49.21 µs | 38.35 µs | 1.28 |
| 64 | 88.22 µs | 70.00 µs | 1.26 |

**The closed form is not yet faster than `Ple` at `k ≤ 64` on a SIMD host.**
The asymptotics are not in question — the closed form is `Θ(k²)` field
operations against elimination's `Θ(k³)` — but `Ple`'s inner loop is 16–64
bytes of GF(2^8) per SIMD instruction, while the closed form's per-entry
products and inversions are scalar. Below the crossover, `k³` SIMD lanes beat
`k²` scalar ops. Switching the inverse core from Montgomery batch inversion to
elementwise table `inv()` (as `srs` does, and as this crate now does) cut the
closed form from ~1.9x slower to ~1.25x, but did not cross over.

Two things move this: (1) vectorizing the closed
form's product and fill loops through `fgf::ops` (the small-matrix kernel
tuning that `SmallMatrix` and the Newton–John tables target), and (2) the
closed form's real production role — fusing coefficient generation with payload
application so the `O(k²)` is amortized across the whole decode rather than
spent materializing a bare inverse. Recorded here rather than asserted as a
passing crossover, per the measure-don't-reason rule.

## Hybrid solve on an RFC LT-degree-shaped system

`tests/hybrid.rs::rfc_degree_inactivation_scales_with_square_root` uses the
exact `Deg[v]` thresholds from RFC 6330 Table 1, uniformly samples LT row
degrees, and adds `ceil(sqrt(k)) + 8` received rows. The table reports the
maximum over eight deterministic seeds:

| k | max inactive columns `g` | `g / sqrt(k)` |
| ---: | ---: | ---: |
| 10 | 1 | 0.316 |
| 25 | 2 | 0.400 |
| 50 | 2 | 0.283 |
| 100 | 2 | 0.200 |
| 250 | 7 | 0.443 |
| 500 | 8 | 0.358 |
| 1000 | 13 | 0.411 |

The measured worst constant is **0.443**. This fixture isolates the LT-degree
schedule; it is not a claim that the crate constructs the RFC precode matrix.

`benches/hybrid.rs`, Intel Core Ultra 7 258V, rustc 1.93, `cargo bench --bench
hybrid`. The `k = 1000` system has 1040 rows, RFC Table 1 LT degrees, and
1024-byte GF(2^8) symbol payloads. `hybrid` reuses its warmed workspaces;
`dense_ple` clones the same coefficient matrix, decomposes it, and solves the
same right-hand side. Median estimates from three interleaved Criterion runs:

| trial | `Hybrid` | dense `Ple` | dense / hybrid |
| ---: | ---: | ---: | ---: |
| 1 | 6.86 ms | 8.09 ms | 1.18 |
| 2 | 6.61 ms | 7.39 ms | 1.12 |
| 3 | 5.51 ms | 9.17 ms | 1.66 |
| maximum | 6.86 ms | 9.17 ms | 1.34 |

Every run favored `Hybrid`; the conservative paired-run ratio is **1.12x**,
while the required maximum-of-three estimates give **1.34x**.


## Elimination and dispatch tuning

Measurements below used an Intel Core Ultra 7 258V, rustc 1.93.0,
`v3_gfni_crypto`, Criterion 0.8.2, and `taskset -c 2`. The retained tuning
benchmarks are in `benches/tuning.rs`; candidates remain callable behind
`internals` so every production dispatch has a direct A/B twin. Unless a table
says otherwise, times are Criterion median estimates and the decision was
unchanged across three pinned runs.

### Dense panel width

The unblocked twin is panel width one. The table shows a representative
boundary run; the production decision uses the worse median from the three
runs.

| Field | Order | width 1 | width 64 | Retained |
| --- | ---: | ---: | ---: | --- |
| GF(2^8) | 32 | 2.60 µs | 2.24 µs | width 64 |
| GF(2^8) | 64 | 4.92 µs | 6.09 µs | width 1 |
| GF(2^16) | 32 | 15.40 µs | 10.62 µs | width 64 |
| GF(2^16) | 64 | 59.54 µs | 41.58 µs | width 64 |
| GF(2^16) | 128 | 246.17 µs | 246.26 µs | width 1 |

Result: on the measured GFNI backend, use width 64 through order 32 for
one-byte fields and through order 64 for two-byte fields. Every other shape,
field width, and backend keeps width one. The byte-for-byte twin test is also
run under forced `v3_gfni_crypto`, `v3`, `v2`, `v1`, and `scalar` backends.

### Newton–John trailing update

One 256-entry multiplication table is built per pivot row and reused across
the trailing submatrix. The values below are the maximum medians from three
runs:

| GF(2^8) order | blocked AXPY | Newton–John | table / AXPY |
| ---: | ---: | ---: | ---: |
| 128 | 11.88 µs | 18.72 µs | 1.58 |
| 256 | 48.38 µs | 50.29 µs | 1.04 |
| 512 | 198.74 µs | 203.94 µs | 1.03 |
| 1024 | 913.29 µs | 952.74 µs | 1.04 |

The table path never won, so production keeps the AXPY update. The candidate
stays behind `internals` as a reproducible rejection, not dormant dispatch.

### GF(2) M4RI slab

`bits::Ple` builds a 256-row XOR table from eight pivots. Three pinned
boundary runs:

| Trial | plain 64 | M4RI 64 | plain 128 | M4RI 128 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.28 µs | 14.85 µs | 64.73 µs | 57.91 µs |
| 2 | 10.19 µs | 14.58 µs | 64.87 µs | 57.84 µs |
| 3 | 10.18 µs | 14.76 µs | 65.41 µs | 57.90 µs |

The table loses at 64 and wins at 128 in every run, so the crossover is 128
rows/columns. `tests/bits_ple.rs` proves table and plain decompositions
byte-identical.

### Compact `SmallMatrix`

`small_matrix` compares construction, factorization, and a 1024-byte GF(2^8)
right-hand-side solve. Both paths receive the same full-rank matrix and include
their own construction. Times below are the three pinned medians:

| Order | `SmallMatrix` runs | `Ple` runs | worst small / `Ple` |
| ---: | --- | --- | ---: |
| 4 | 0.231 / 0.226 / 0.227 µs | 0.362 / 0.354 / 0.356 µs | 0.64 |
| 8 | 0.981 / 1.005 / 1.001 µs | 1.251 / 1.269 / 1.258 µs | 0.80 |
| 16 | 4.510 / 4.558 / 4.428 µs | 5.435 / 5.476 / 5.405 µs | 0.83 |
| 32 | 17.789 / 18.434 / 17.990 µs | 20.697 / 21.461 / 21.044 µs | 0.86 |
| 48 | 41.785 / 42.035 / 41.856 µs | 55.815 / 55.459 / 54.682 µs | 0.77 |
| 64 | 74.802 / 77.770 / 77.870 µs | 98.256 / 100.722 / 100.144 µs | 0.78 |

Every tested order from 4 through the supported maximum of 64 favored the
compact path in all three runs. The hybrid solver therefore uses it for
full-rank square residuals over one-byte fields through order 64; wider,
rectangular, deficient, and wider-field blocks use `Ple`.

### Optional Rayon row updates

`benches/parallel.rs` compares the same `fgf::ops::mul_add` work in fixed
one-thread and eight-thread Rayon pools. A representative run:

| Contiguous bytes | serial | eight threads | Outcome |
| ---: | ---: | ---: | --- |
| 512 KiB | 14.83 µs | 18.76 µs | serial |
| 1 MiB | 29.94 µs | 24.43 µs | unstable across runs |
| 2 MiB | 99.86 µs | 57.03 µs | parallel |
| 4 MiB | 315.56 µs | 124.58 µs | parallel |
| 8 MiB | 667.24 µs | 327.66 µs | parallel |

Three boundary runs made 2 MiB the first repeatable win; 1 MiB crossed within
run-to-run noise and 512 KiB lost. The `parallel` feature is therefore
off by default and delegates only contiguous updates of at least 2 MiB.

## Same-host library comparison

`benches/competitors.rs` times rank of the same deterministic matrices against
the installed current libraries: FLINT 3.6.0, M4RI 20260122, M4RIE 20250128,
and FFLAS-FFPACK 3.6.0. Every path constructs its owned native matrix from the
same immutable input inside the timed region, so allocation and import costs
are included. The GF(2^8) comparisons use the modulus `0x11B`, matching `fgf`;
the differential tests check FLINT and M4RIE against `gfm` before either enters
the benchmark record.

GF(2), three pinned medians:

| Order | `gfm` | M4RI | FFLAS-FFPACK |
| ---: | --- | --- | --- |
| 128 | 57.7 / 57.5 / 58.1 µs | 60.9 / 61.7 / 61.7 µs | 880.1 / 880.2 / 882.1 µs |
| 256 | 305.9 / 305.9 / 312.9 µs | 315.2 / 315.1 / 316.5 µs | 4536.6 / 4557.7 / 4474.9 µs |
| 512 | 1224.0 / 1226.7 / 1228.5 µs | 1392.8 / 1413.7 / 1430.4 µs | 28908.7 / 29023.9 / 29175.4 µs |

GF(2^8), three pinned medians:

| Order | `gfm` | M4RIE | FLINT |
| ---: | --- | --- | --- |
| 128 | 0.127 / 0.128 / 0.130 ms | 0.433 / 0.429 / 0.431 ms | 19.675 / 19.600 / 19.798 ms |
| 256 | 0.560 / 0.554 / 0.575 ms | 1.177 / 1.190 / 1.181 ms | 104.160 / 103.681 / 105.508 ms |
| 512 | 2.689 / 2.704 / 2.757 ms | 4.743 / 4.817 / 4.855 ms | 549.184 / 545.504 / 552.704 ms |

FLINT's generic `fq_nmod` import dominates this end-to-end measurement: each
byte is expanded through eight polynomial-coefficient setter calls before
`fq_nmod_mat_lu` runs. That is the public owned-matrix construction path this
harness can compare fairly, not a claim that FLINT's elimination kernel alone
is 100–200 times slower.

These are same-host implementation measurements, not claims about every
machine or workload. Missing native libraries produce a loud skipped test and
omit the corresponding benchmark instead of silently substituting a mock.

## Consumer cutover checks

The first direct-consumer cutover was measured on 2026-08-08 on the same host,
with each benchmark pinned to CPU 2. Baselines are detached worktrees at each
consumer's pre-cutover `main`; migrated builds use the same benchmark inputs and
optimized profile. A negative delta is faster. These are end-to-end caller
checks, not dispatch thresholds.

| Consumer / case | Baseline | Migrated | Delta |
| --- | ---: | ---: | ---: |
| `mix-dpc` solver, 4 rows | 1.6790 µs | 1.4962 µs | -10.89% |
| `mix-dpc` solver, 12 rows | 7.3403 µs | 6.5073 µs | -11.35% |
| `mix-dpc` solver, 32 rows | 39.882 µs | 32.350 µs | -18.89% |
| `srs` direct decode, `k64_m32` | 78.783 µs | 74.240 µs | -5.77% |
| `ccrlnc` systematic decode | 243.30 ns/item | 231.55 ns/item | -4.83% |
| `ccrlnc` 10% loss decode | 939.17 ns/item | 979.98 ns/item | +4.35% |
| `ccrlnc` systematic encode | 692.74 ns/item | 697.03 ns/item | +0.62% |
| `ccrlnc` systematic recode | 49.53 ns/item | 48.14 ns/item | -2.81% |
| `ccrlnc` 10% loss recode | 784.05 ns/item | 720.66 ns/item | -8.08% |
| `ccrlnc` dense recode | 1566.32 ns/item | 1621.89 ns/item | +3.55% |
| `cafft` GF(16) forward, `p128_r64` | 890.65 ns | 881.68 ns | -1.01% |
| `cafft` GF(16) inverse, `p128_r64` | 981.39 ns | 964.22 ns | -1.75% |
| `cafft` GF(16) derivative, `p128_r64` | 829.88 ns | 773.33 ns | -6.81% |
| `cafft` GF(8) forward, `p128_r64` | 750.55 ns | 747.01 ns | -0.47% |
| `cafft` GF(8) inverse, `p128_r64` | 787.15 ns | 790.50 ns | +0.43% |

`gs-engine`'s custom harness reports one elapsed total per case rather than a
sample distribution, so the table below uses the slower result from three
pinned runs, matching this file's measurement rule. The unchanged Kötter
interpolation path is the 1.00x control.

| Module case | Baseline | Migrated | Raw delta |
| --- | ---: | ---: | ---: |
| GF(8), 31 points | 1.398 ms | 1.533 ms | +9.66% |
| GF(8), 63 points | 12.505 ms | 12.697 ms | +1.53% |
| GF(8), 255 points | 72.631 ms | 72.021 ms | -0.84% |
| GF(16), 31 points | 2.434 ms | 2.369 ms | -2.64% |
| GF(16), 63 points | 20.199 ms | 18.559 ms | -8.12% |
| GF(16), 255 points | 119.109 ms | 114.351 ms | -3.99% |

The GF(8), 31-point control moved from 3.036 ms to 3.236 ms (+6.59%) over the
same runs. Relative to that control, the module path moved +2.88%; no
control-normalized regression crossed 5%.
## Hybrid sparse-phase scheduler: from quadratic scans to indexed scheduling

`benches/rfc_scale.rs`, added with this change. Systems are RFC 6330-shaped:
LT-degree binary rows over `W` columns plus overhead, optionally `H` dense
GF(256) field rows spanning every column (the HDPC band), with the trailing
`H` columns pre-inactivated. `iter_custom` single-shot timing; criterion
medians; release profile; development host (Core Ultra 7 258V).

The baseline (`main` before this change) scheduled with four per-iteration
full scans: an active-weight recount over all live entries, a minimum-weight
scan over all rows, a weight-two edge rebuild over all rows, and a
row-per-row elimination probe. Each is `O(m)` or worse per pivot, so
solves grew quadratically-to-cubically and a `K = 56403` RaptorQ
intermediate-symbol solve took ~300 s end to end.

Strategies, measured cumulatively (each row includes the previous ones):

| Strategy | lt_only 1k | lt_only 5k | lt_only 20k | eager 4k | deferred 4k | deferred 56403 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (`main`) | 4.74 ms | 301 ms | 5.73 s | 3.33 s | — | — |
| A: incremental weights | 2.67 ms | 123 ms | 2.92 s | 3.28 s | — | — |
| A+C: deferred dense rows | 2.66 ms | 122 ms | 2.93 s | 3.28 s | 82.2 ms | 26.1 s |
| A+C+B: bucketed selection | 1.87 ms | 37.4 ms | 1.83 s | 3.55 s | 73.3 ms | 17.3 s |
| A+C+B+D: column index (final) | 0.334 ms | 2.02 ms | 12.1 ms | 3.73 s | 42.7 ms | 655 ms |

- **A — incremental weights**: weights are initialized once in
  `prepare_work`, recomputed inside the merge that rewrites a row, and
  decremented on inactivation. Kills the per-iteration recount: ~2x on
  LT-only systems; no effect on dense-row systems (merge-bound).
- **C — deferred dense rows** (`push_deferred_field_row`): dense rows skip
  the sparse phase entirely and are released with their pivoted columns
  substituted out in one pivot-time-ordered pass. The eager-vs-deferred
  twins stay compiled side by side in the bench. Deferred beats eager by
  6.7x at 500 columns, 40x at 4k, and turns max-K from hours-extrapolated
  into 26 s.
- **B — bucketed weight queues**: every live row sits in the queue for its
  current weight; minimum-weight selection and the weight-two edge rebuild
  become queue scans instead of row scans.
- **D — column-to-row index**: a pivot's elimination visits the rows listed
  under its column (initial supports plus merge-time additions, cancellations
  tolerated as stale entries filtered by the coefficient lookup, duplicates
  suppressed by a per-pivot generation counter) instead of probing every row.
  LT-only becomes near-linear (5.73 s to 12.1 ms at 20k columns).

Controls: `raptorq_shaped_k1000/dense_ple` 7.56 ms to 7.55 ms (1.00x);
`competitors_gf8_rank/gfm/128` 125.9 us to 126.7 us (1.01x). The
pre-existing `raptorq_shaped_k1000/hybrid` bench improves 5.01 ms to
0.560 ms (9.0x).

Regression: the eager dense-band path pays ~12% (3.33 s to 3.73 s at 4k
columns) for index bookkeeping on rows that contain nearly every pivot
column anyway. That path is superseded by deferral for dense rows; sparse
systems are 14x-470x faster.

Downstream, `raptor-q` (consumer, same host): encoder preparation at
`K = 56403`, `T = 64` improved 297 s to 1.9 s (~155x); `K = 1000` from
28.9 ms to 5.1 ms; steady-state repair generation unchanged at 80 ns per
symbol. Its frozen differential fixtures pass byte-identically before and
after, which is the answer-invariance proof for the deferral release.
