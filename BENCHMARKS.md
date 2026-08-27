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

### GF(2) domain on `fgf::bits` (2026-08-24)

The bit domain rebased from private `u64` word loops onto `fgf::bits`'
packed-byte surface (fgf 0.7), the same cutover the dense domain already
had through `fgf::ops`. The private `xor_range`/`xor_all`/`clear_prefix`
word loops are gone; the elimination calls the checked public kernels,
and `locate_pivot` scans the one-panel window (at most two bytes at the
production width) byte-wise. A/B against the pre-cutover code
(`--baseline precutover`, three interleaved runs, same host):

| Shape | pre-cutover | on `fgf::bits` | change |
| --- | ---: | ---: | ---: |
| plain 64 | 10.15 µs | 13.11 µs | +29% |
| m4ri 64 | 15.44 µs | 21.61 µs | +40% |
| plain 128 | 62.57 µs | 85.14 µs | +36% |
| m4ri 128 | 54.52 µs | 79.52 µs | +46% |
| plain 256 | 415.26 µs | 510.93 µs | +23% |
| m4ri 256 | 282.62 µs | 348.66 µs | +23% |
| plain 512 | 1.846 ms | 2.267 ms | +23% |
| m4ri 512 | 1.121 ms | 1.367 ms | +22% |
| plain 1024 | 7.772 ms | 9.556 ms | +23% |
| m4ri 1024 | 4.585 ms | 5.570 ms | +22% |

Cost anatomy (perf, plain 1024): ~30% of samples inside
`fgf::bits::xor_range` — the surface's length/range/coverage contract
checks and sub-word mask arithmetic, not word assembly. Two fgf-side
kernel fixes landed with this cutover and are recorded in fgf's
`BENCHMARKS.md`: the masked range kernels run their fully-live interior
through the dispatched bulk XOR (5.5–7.4x on fgf's own range bench in the
cache tiers), and short buffers take an inlined portable path under the
dispatched call boundary. Without those the regression was +38–68%. The
M4RI crossover stays at 128 (the table still loses at 64 and wins at 128);
decomposition output is byte-identical, proven by the existing
cross-domain differential and the M4RI/FFLAS-FFPACK oracles.

**Second round — prepared ranges.** The per-row cost was mostly per-call
surface: an elimination applies the *same* range to many rows, so fgf
gained the prepare/apply split `ops` already had for coefficients
(`bits::RangeXor` + `xor_range_with`; 3.6x the one-shot form per call on
fgf's own short-row bench, see fgf's `BENCHMARKS.md`). `factor_panel`
prepares once per pivot, `bulk_update` and the M4RI table once per panel,
and every row applies. Same baseline, same host:

| Shape | pre-cutover | prepared ranges | change |
| --- | ---: | ---: | ---: |
| plain 64 | 10.15 µs | 10.68 µs | +5% |
| plain 128 | 62.57 µs | 69.48 µs | +11% |
| plain 256 | 415.26 µs | 454.96 µs | +9% |
| plain 512 | 1.846 ms | 1.955 ms | +6% |
| plain 1024 | 7.772 ms | 8.199 ms | +5% |
| m4ri 128 | 54.52 µs | 64.20 µs | +18% |
| m4ri 256 | 282.62 µs | 302.81 µs | +7% |
| m4ri 512 | 1.121 ms | 1.191 ms | +6% |
| m4ri 1024 | 4.585 ms | 4.824 ms | +5% |

At production sizes (the m4ri path, ≥128 rows/columns) the standing cost
is +5–7%; the plain path's small-n outlier is the byte-wise pivot scan,
and the m4ri small-n outlier additionally pays the table-build copies.
What remains is structural to a checked byte-typed surface: the per-apply
coverage check, bounds-checked byte reads, and `locate_pivot`'s two byte
loads per row where the private code loaded one word. Closing that needs
an unchecked kernel escape hatch in fgf or `unsafe` word views in gfm;
both are outside the crates' rules.

### GF(2) selector read closes the plain path (2026-08-24)

Follow-up to the cutover above. The +5–7% standing cost is dominated by the
`fgf::bits` per-apply surface, but part of it is gfm-side: the trailing
update tested each pivot column with a bounds-checked `BitMatrix::get` — one
`region()` reslice and map indirection per bit — up to eight per trailing
row. `BitMatrix::row_selector` reads the whole L-factor selector for a panel
in a single masked byte-window load (at most two live bytes at the
production `SLAB_WIDTH` of eight), and the plain trailing update walks the
set bits with `trailing_zeros` instead of re-reading each column. The M4RI
key extraction reads the same way. Output is byte-identical (the plain/table
differential and the cross-domain oracle both hold). A/B against the
`prepared ranges` baseline above (`--baseline pre`, three interleaved runs,
same host; the GF(2^8) `dense_dispatch` group is the unchanged 1.00x
control):

| Shape | prepared ranges | selector read | change |
| --- | ---: | ---: | ---: |
| plain 128 | 68.83 µs | 47.83 µs | −31% |
| plain 256 | 453.34 µs | 311.53 µs | −31% |
| plain 512 | 1.949 ms | 1.467 ms | −25% |
| plain 1024 | 8.204 ms | 6.301 ms | −23% |
| m4ri 128 | 64.13 µs | 63.03 µs | −2% |
| m4ri 256 | 304.17 µs | 301.26 µs | −1% |
| m4ri 512 | 1.190 ms | 1.196 ms | +1% |
| m4ri 1024 | 4.828 ms | 4.818 ms | 0% |

At this stage the selector read is a small fraction of an m4ri trailing row:
one wide `xor_range_with` dominates, so the table path is unmoved. The plain path,
which paid the per-bit `get` on every pivot column of every trailing row,
drops by a quarter to a third — enough that it beats the table out to a
larger size and moves the crossover (below). Relative to the pre-cutover
private word loops, the plain path is now faster (plain 128 47.83 µs vs
pre-cutover 62.57 µs, −24%); the residual +5–7% is confined to the m4ri
path at ≥224, where the dominant cost stays inside `fgf::bits`.

### GF(2) prepared backend closes the remaining cutover cost (2026-08-24)

The remaining M4RI cost was one level lower: `fgf::bits::RangeXor` prepared
the byte window and masks but resolved the process XOR backend again on every
`xor_range_with`. It also peeled both end bytes as masked scalars even when a
byte-aligned suffix made both masks `0xFF`. `RangeXor` now captures the
resolved backend once, and fgf's apply kernel keeps fully-live end bytes in
the bulk XOR. Its retained old/new twin is 1.17–1.21x faster per call on the
16–128-byte row shapes.

Consumer A/B used three paired runs, each pinned fgf immediately followed by
the local optimized fgf, over the production M4RI sizes. Improvements ranged
from 3.6–10.6%; the weakest paired result was −3.6%, −4.2%, and −4.2% at
orders 256, 512, and 1024. More importantly, the slowest optimized medians
across those runs versus the private-loop pre-cutover production baseline are:

| Order | pre-cutover | optimized fgf | residual |
| ---: | ---: | ---: | ---: |
| 256 | 282.62 µs | 291.54 µs | +3.2% |
| 512 | 1.121 ms | 1.133 ms | +1.0% |
| 1024 | 4.585 ms | 4.604 ms | +0.4% |

Order 128 already runs the selector-optimized plain path at 47.8 µs, 12%
faster than the pre-cutover M4RI path. The reported 5–10% GF(2) cutover
penalty is therefore closed: production sizes are faster at 128 and within
0.4–3.2% at larger orders, below the run-to-run host variance.


### GF(2) M4RI slab

`bits::Ple` builds a 256-row XOR table from eight pivots. The selector read
and fgf prepared-backend changes moved the boundary; paired forced-path runs
across the final transition (plain / M4RI, µs):

| n | plain | M4RI | winner |
| ---: | ---: | ---: | --- |
| 128 | 47.2 | 65.1 | plain (M4RI +38%) |
| 160 | 88.3 | 94.9 | plain (M4RI +7.5%) |
| 192 | 141.5 | 148.5 | plain (M4RI +4.9%) |
| 224 | 213.4 | 210.1 | M4RI (+1.5%) |

The table loses through 192 and wins from 224 up, so the crossover is 224
rows/columns (`M4RI_CROSSOVER`). `tests/bits_ple.rs` proves table and plain
decompositions byte-identical, so the constant is a pure performance
boundary.

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
| A+C+B+D: column index | 0.334 ms | 2.02 ms | 12.1 ms | 3.73 s | 42.7 ms | 655 ms |
| +E: coefficient-free binary rows | 0.268 ms | 1.66 ms | 10.0 ms | — | — | — |
| +F: edge cache, folded weight, batched release (final) | 0.268 ms | 1.66 ms | 10.0 ms | 9.8 ms (500c) | 23.3 ms | 278 ms |

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

## Second round: coefficient-free binary rows, edge cache, folded weights, batched release

Profiled against the consumer (`raptor-q` `K = 56403`, `T = 64`; phase
timers inside `run_into`): the sparse loop held ~0.9 s, the deferred
release ~0.5 s, everything else ~0.1 s. Four more strategies, measured
cumulatively on top of the first round:

- **E — coefficient-free binary rows**: a binary row's coefficients are
  implicit units (`coeffs` empty); a field operation widens the row once
  by materializing them. Binary/binary merges with a unit factor — the
  entire GF(2) peeling majority, `widenings = 0` in the RaptorQ solve —
  degenerate to a support XOR with no coefficient traffic or field
  multiplies.
- **Edge cache**: each weight-two row's edge is cached and re-extracted
  only when its support or weight changed; most weight-two rows are
  stable across consecutive peeling steps. The previous code rescanned
  every weight-two row's full support every iteration (164 M entry scans
  at `K = 56403`).
- **Folded weight maintenance**: the merged row's active weight is
  counted inside the merge walk (91 M second-pass scans removed), and
  inactivation decrements use the column-to-row index with a generation
  guard instead of scanning all rows (8 M row scans removed).
- **Batched release**: deferred rows are substituted in lane groups of
  sixteen — one pass over the pivots in time order with a per-column
  lane bitmask — so the frozen pivot rows stream once per group instead
  of once per deferred row, and the ordering heap disappears. Release
  time fell from ~0.5 s to ~0.2 s at `K = 56403`.

| Case | First round | Second round |
| --- | --- | --- |
| `rfc_scale` deferred 56403 | 655 ms | 278 ms |
| `raptor-q` solve K=56403 | 1.97 s | 1.37 s |
| `raptor-q` prepare K=56403 | 1.91 s | 1.36 s |
| `raptor-q` prepare K=1000 | 5.1 ms | 4.0 ms |
| `raptor-q` decode K=1000 | 5.2 ms | 4.1 ms |

`raptor-q`'s frozen differentials and the `cberner/raptorq` interop
checks (byte-identical repairs through `K = 56403`, cross-decode both
directions) pass unchanged: every strategy is schedule-only. The head-to-
head gap narrows to ~10x at max K (1.52 s vs 0.131 s prepare); the
remaining cost is the sparse loop's u32-per-column GF(2) representation —
the reference implementation runs packed 64-column words, which is the
next order of magnitude and a representation project, not a patch.

A lane-group regression test covers systems with more than sixteen
deferred rows (the bench shapes caught the original cap).

## Third round: packed frozen rows in the sparse domain (2026-08-26)

The round-two diagnosis attributed the remaining max-`K` gap to the
u32-per-column GF(2) representation. This round acted on it: each packed
(binary) row now splits its support into `cols`, the columns still active in
the schedule, and `frozen`, bit-packed words over the inactivated columns
keyed by a global frozen ordinal — the same split the reference PI solvers
draw between sparse lists and a dense trailing block. Freezing moves entries
out of the lists once, at inactivation time; merges then XOR whole words.
Field-valued rows keep flat lists and pay list merges (a bit cannot carry a
GF(2^8) coefficient); widening materializes a packed row exactly once.

### Where the time actually was

Phase timers inside `run_into` on the synthetic `lt_hdpc_deferred/56403`
shape (Core Ultra 7 258V, rustc 1.93, pinned CPU, medians of eight runs)
corrected the cost model before any tuning:

| Phase | share |
| --- | ---: |
| redundant-row verification | ~38% |
| release (sweep, deferred substitution) | ~15% |
| dense assembly + solves | ~14% |
| sparse loop | ~15% |
| prepare_work | ~13% |
| back-substitution + kernel lift | ~6% |

The sparse loop the representation targets was already only ~15% — the
column-index scheduling of rounds one and two made merges rare — and the
single largest line item is consistency verification of dependent rows,
which walks *input* rows (~1018 verified rows carrying ~5900 accumulated
entries each at this shape, ~6.0M coefficient-payload operations per solve).
Verification is untouched by any working-row representation.

### Consumer paired runs (real RFC systems, Gf8D)

Three interleaved criterion medians per state, same host, pinned CPU;
worst median shown. `raptor-q` builds against this crate by path, so
prepare/decode/repair measure end-to-end consumer impact.

| Case | flat list | packed frozen | change |
| --- | ---: | ---: | ---: |
| `raptor-q` prepare K=56403 | 1.4995 s | 0.5952 s | −60% |
| `raptor-q` prepare K=1000 | 4.454 ms | 2.167 ms | −51% |
| `raptor-q` decode k=10 | 24.78 µs | 22.23 µs | −10% |
| `raptor-q` decode k=100 | 229.6 µs | 148.9 µs | −35% |
| `raptor-q` decode k=1000 | 4.514 ms | 2.218 ms | −51% |
| `raptor-q` repair t=64 | 87.9 ns | 86.4 ns | −2% |
| `raptor-q` repair t=1024 | 192.8 ns | 171.7 ns | −11% |

Repair generation does not enter the solver after preparation; its shift is
code-layout noise on an untouched path. The real-system wins come mostly
from tuple-structured G_ENC supports, which fill far more aggressively than
uniform random rows and had been paying full-length list merges.

### Synthetic shapes (Gf8B)

Same protocol against this crate's own benches:

| Case | flat list | packed frozen | change |
| --- | ---: | ---: | ---: |
| `rfc_scale` lt_only 1000 | 293 µs | 267 µs | −9% |
| `rfc_scale` lt_only 5000 | 1.806 ms | 1.698 ms | −6% |
| `rfc_scale` lt_only 20000 | 10.27 ms | 10.05 ms | −2% |
| `hybrid` k1000 hybrid | 472 µs | 440 µs | −7% |
| `hybrid` k1000 dense_ple (control) | 8.068 ms | 8.066 ms | 1.00x |
| `rfc_scale` lt_hdpc_deferred 500 | 674 µs | 576 µs | −15% |
| `rfc_scale` lt_hdpc_deferred 4000 | 26.11 ms | 25.62 ms | −2% |
| `rfc_scale` lt_hdpc_deferred 56403 | 293.7 ms | 305.4 ms | +4% |

The one standing cost is the synthetic max-K shape. Its band is wider than
the real system's (108 pre-inactivated columns versus H=190 over a column
space the deferred rows span), and its uniform-random rows peel so cleanly
that sequential iteration over pivot-row frozen bits during deferred release
is slightly slower than the contiguous u32 walk it replaced. On the real
workload that same scale runs 2.5x faster, which is the trade this record
keeps. Answers are byte-identical: the schedule depends only on active-set
weights, which are unchanged, and the existing eager/deferred, cross-domain,
and exact-value differentials pass unchanged.

## Fourth round: verify dependent rows in reduced form (2026-08-26)

The phase-timer profile above put redundant-row verification at ~38% of the
max-K solve, walking input supports (~6.0M coefficient-payload operations on
`lt_hdpc_deferred/56403`, about 90% of it released HDPC rows re-walking
their L-wide original supports). The verifier now checks each dependent row
in the cheapest algebraically equivalent form:

- **Released deferred rows** evaluate their rebuilt coefficients over the
  inactive columns against their substituted right-hand side. Release
  substitutes pivots out of coefficients and collects the matching payload
  factors, so the reduced equation holds exactly when the original one does;
  the L-wide input walk disappears.
- **Binary rows** evaluate as one XOR of the selected value rows — unit
  coefficients need no lookup and no multiply (`ops::add_assign`).
- Field rows keep the entry-by-entry input-support evaluation.

Deferral semantics and `row_ops` counting are untouched: the sparse-phase
log still replays only to needed rows, and surplus sparse rows still verify
from their inputs. Per-row verdicts are equivalent identities, so
`Inconsistent { row }` names the same first failing row.

Synthetic shapes, same protocol as round three (packed-frozen state vs this
round; HEAD baseline shown for reference):

| Case | HEAD | packed | reduced verify |
| --- | ---: | ---: | ---: |
| `rfc_scale` lt_hdpc_deferred 500 | 674 µs | 576 µs | 456 µs |
| `rfc_scale` lt_hdpc_deferred 1000 | 1.757 ms | 1.751 ms | 1.152 ms |
| `rfc_scale` lt_hdpc_deferred 2000 | 6.46 ms | 6.46 ms | 3.88 ms |
| `rfc_scale` lt_hdpc_deferred 4000 | 26.11 ms | 25.62 ms | 14.84 ms |
| `rfc_scale` lt_hdpc_deferred 56403 | 293.7 ms | 305.4 ms | 203.9 ms |
| `rfc_scale` lt_only 20000 | 10.27 ms | 10.05 ms | 9.95 ms |
| `hybrid` k1000 hybrid | 472 µs | 440 µs | 440 µs |
| `hybrid` k1000 dense_ple (control) | 8.068 ms | 8.066 ms | 7.915 ms |

Against the pre-round baseline, max-K drops 31% and the mid-range shapes
34–43%; the control sits inside run-to-run noise. The consumer is unmoved:
encoder and decoder systems carry few redundant equations (prepare K=56403
600 ms vs 595 ms), so its remaining cost stays in the merge machinery this
record's round-three tables already describe.

## Fifth round: release in the inactive column space (2026-08-26)

Consumer profiling after round four put 42.8% of `raptor-q` prepare
K=56403 inside the deferred-release substitution. Counters added to the
release loop showed two shapes: on the real system the substitution walks
touch ~11.0M entries with a 16-wide lane inner loop over an `n`-wide
(900 KB) accumulator; on the synthetic max-K shape the same loop records
~6.0M right-hand-side op tuples and rescans all pivots once per group
(seven groups for 108 deferred rows).

The rewrite rests on one invariant: pivot rows carry nothing outside their
pivot column and the inactive set, so substitutions never write a pivot
column, and every walked non-pivot column is inactive (now pinned by a
debug assertion). That makes each seeded coefficient at a pivot column
immutable until its own pivot fires — factors need no evolution tracking —
and lets the accumulation live in a `g`-wide accumulator indexed by
inactive ordinal:

- the lane store becomes read-only seed; emission adds seed to accumulator;
- the per-entry inner loop iterates a hoisted live-lane list instead of
  re-testing the mask bit for every lane of every entry;
- the pivot's own column is skipped instead of written and discarded;
- dead mask bookkeeping on substituted-in columns is gone;
- groups widen from sixteen lanes to sixty-four (`RELEASE_LANES`),
  dividing group-rescan and walk overhead by four where bands are wide.

Same protocol as rounds three and four (verify-fix state vs this round):

| Case | verify fix | release fix | change |
| --- | ---: | ---: | ---: |
| `rfc_scale` lt_hdpc_deferred 500 | 456 µs | 392 µs | −14% |
| `rfc_scale` lt_hdpc_deferred 1000 | 1.152 ms | 1.034 ms | −10% |
| `rfc_scale` lt_hdpc_deferred 2000 | 3.88 ms | 3.50 ms | −10% |
| `rfc_scale` lt_hdpc_deferred 4000 | 14.84 ms | 13.57 ms | −9% |
| `rfc_scale` lt_hdpc_deferred 56403 | 203.9 ms | 177.7 ms | −13% |
| `raptor-q` prepare K=56403 | 603 ms | 486 ms | −19% |
| `raptor-q` prepare K=1000 | 2.153 ms | 1.973 ms | −8% |
| `raptor-q` decode k=1000 | 2.243 ms | 2.098 ms | −6% |

Cumulative against the session-start baseline in this file's round-three
table: consumer prepare at max K falls from 1.4995 s to 0.486 s (−68%),
decode K=1000 from 4.514 ms to 2.098 ms, and the synthetic max-K solve
from 293.7 ms to 177.7 ms. Answers remain byte-identical under the
eager/deferred differentials, including a dense band wider than one lane
group. The remaining prepare profile now leads with the sparse-phase merge
walks and payload replay rather than any single dominant phase.

### Rejected: unit-coefficient fast paths and component-tracking variants

Three follow-up micro-candidates were built and measured against the
release-fix state, and none survived:

- **Unit spread in release** (skip the per-entry multiply when the pivot
  row is binary) and **XOR back-substitution** (`add_assign` for packed
  pivot rows): symbol shares moved as designed (release 27.7% → 22.8%,
  back-substitution 10.6% → 7.2%), but end-to-end totals stayed flat —
  the saved table multiplies reappeared as per-entry kernel-dispatch
  overhead in the XOR path (`xor_impl` grew by roughly what `mul_add`
  lost). The cost is call granularity, not arithmetic.
- **Amortized union-find for the weight-2 tie-break**, both as a
  touched-list reset over the full column arrays and as a compact sorted
  endpoint universe: flat to worse. Measured selection structure on
  `raptor-q` prepare K=56403 is ~800 calls averaging ~8273 edges (~130 on
  the synthetic max-K shape); the sort-per-call variant regressed prepare
  by ~15%. A meaningful win here needs incremental component maintenance,
  which is schedule surgery, not a constant-factor patch.

All three were reverted; the working tree matches the committed release-fix
state.

### Rejected: kernel-dispatched deferred-release substitution (2026-08-26)

Two further candidates for the release inner loop, both measured against the
release-fix state with the same protocol (`rfc_scale` `lt_hdpc_deferred`
500–56403, paired criterion medians, pinned CPU) and both **slower**, so both
were reverted:

- **Per-entry lane-vector AXPY**: the accumulator became a byte slab with the
  64-lane dimension contiguous, dead lanes neutralized by zero factors, and
  the scalar live-lane loop replaced by one dispatched `mul_add` per pivot
  entry. Result: **+0.5% to +6.9%** across the deferred shapes. The per-entry
  kernel dispatch costs more than the 64 inlined table multiplies it replaces
  — the accumulator is L1-resident and the scalar chain pipelines.
- **Per-pivot scattered-matrix kernel**: one `ops::mul_add_matrix_scattered`
  call per fired pivot (touched accumulator rows collected with their
  coefficients, lane-major factor bytes as the single term). Result: **+0.2%
  to +6.1%**. The public wrapper's O(k²) disjointness validation plus
  per-row coefficient preparation inside the kernel outweigh the arithmetic
  saving; a profiled run showed the cost sitting in dispatch and validation
  rather than the multiply itself.

Both confirm round five's diagnosis at lane width 64: the release loop's cost
is call granularity and per-call fixed work, not field arithmetic. A winning
path needs an fgf op shaped as a sequential indexed row-scatter with O(k)
validation and no per-row preparation — recorded as a candidate, not built;
the scalar lane loop stays production.

### Rejected: dropping the weight-two component tie-break (2026-08-26)

Replacing the RFC §5.4.2.2 largest-component selection with plain
bucket-order first-row selection (deleting the edge cache and union-find
entirely). The acceptance tests' `g` bounds held easily — worst
`g/sqrt(k)` moved 0.443 → 0.537 on pure LT shapes, band shapes unchanged —
but the wall clock split by shape family (same protocol, paired medians):

| Shape family | Change |
| --- | --- |
| `rfc_scale` `lt_only` 1k–20k | **+1.0% to +7.7%** (regression) |
| `rfc_scale` `lt_hdpc_deferred` 500–56403 | −2.9% to −12.3% |
| `raptor-q` prepare K=1000 / 56403 | −3.6% / −6.3% |

Pure-LT peeling pays for the component rule with longer chains and a better
merge structure; the banded RaptorQ-like shapes paid the rule's analysis
cost without needing its protection. A generic crate serves both families,
so the component rule stays. Reverted to the committed schedule.

### Scoped out on recorded numbers: dense-only small systems, gather replay

Two more candidates from the same consumer-driven list were rejected on this
file's own measurements without building:

- **Dense-only dispatch for small systems** (skip the sparse phase below a
  column threshold): the compact-`SmallMatrix` record prices order-32
  construction + factorization + a 1024-byte solve at ~18 µs — already the
  whole of the consumer's `prepare` at `K = 10` (~23 µs) — and the
  competitor record prices a GF(2^8) rank at order 128 at ~127 µs against a
  ~90 µs solve share at `K = 100`. A dense-only path loses at both ends of
  the small range; the small-`K` consumer gap is plan caching and per-symbol
  tuple derivation on the codec side, not solver dispatch.
- **Gather-fused deferred-log replay**: the consumer profile caps all
  payload kernel time (`xor_impl`, `mul_add_affine_impl`, `xor`) at ~3.6%
  of max-`K` prepare; fusing the per-destination ops through the existing
  gather kernels bounds the win under half of that — below the session's
  noise band.

What remains for the consumer gap are the two structural projects already
named in earlier records: the split-domain dense phase (packed-word GF(2)
elimination for the binary majority with a field fix-up for the HDPC band —
the reference implementation's core advantage) and an incremental-solve API
(warm-started re-solves for the decoder's repeated attempts). Both are
representation and API projects with their own measured changes, not
constant-factor patches.

### Scoped out by measurement: the split-domain dense phase (2026-08-26)

Phase timers inside `run_into` (env-gated, temporary) over the *real* consumer
systems — `raptor-q` prepare and 5%-loss decode at `K = 56403`, `T = 64`,
pinned CPU, medians of five — put the dense phase far below the share that
motivated the split-domain design (the ~14% figure dated from round three,
on the synthetic shape, before the round-five release rewrite):

| Phase | prepare share | decode share |
| --- | ---: | ---: |
| release (deferred substitution) | 35–36% | 35–36% |
| sparse loop | 30–31% | 31–32% |
| **back-substitution (per-entry dispatch)** | **20–21%** | **20%** |
| prepare_work | 5–7% | 5–7% |
| deferred replay | 2% | 2% |
| **rank Ple + solve Ple + dense solve** | **~3% (12 ms)** | **~3% (13 ms)** |
| kernel lift + verification | ~0% | ~0% |

`g = 530–558`, `residual ≈ g` at both shapes (the extra received rows are
consumed as pivot rows, not residual rows). A split-domain dense phase —
packed GF(2) elimination of the binary residual majority plus a field fix-up
for the ≤16 released HDPC rows — bounds the win at roughly two-thirds of
12–13 ms even if the dense phase vanished: **under the session's ±3–5% noise
band, unkeepable by the measured-change rule.** Not built; the phase-timer
table redirects attention to the back-substitution loop (below).

### Rejected: gathered XOR back-substitution, including a new fgf gather op

The phase table made the back-substitution loop (~20%, ~85 ms) the top
actionable target. Its per-entry shape — one dispatched `mul_add` per
support entry — looked dispatch-bound, matching round five's call-granularity
diagnosis. Two layers were built and measured:

1. **fgf `add_assign_gather`**: a unit-coefficient gather (one backend
   resolve, tight per-source XOR loop) wired as a `FieldKernels` override
   for every binary field, with oracle and panic tests — used by gfm to
   substitute each wide binary pivot through one gathered call (chunked
   source arrays, owned scratch, zero steady-state allocation).
2. **gfm consumption**: binary pivot rows with 16–1024-entry supports
   substitute via the gather; everything else keeps the per-entry path.

Result: **flat.** The gather fired on ~97% of pivots (~270 sources each,
frozen-word-dominated) and back-substitution stayed at 82–86 ms — medians
85.6 → 85.4 ms across adjacent instrumented runs, with end-to-end walls
unchanged. The measured cost model explains it: the per-entry loop was
already at ~5.7 ns per source — the dispatched `mul_add` for a one-
coefficient 64-byte row costs ~4 ns after inlining, not the ~12 ns
assumed — while the gather's staging (frozen-bit enumeration, column
collection, source-slice construction, chunk-array initialization) adds
~3–4 ns per source, exactly replacing the dispatch it saved. Enumeration,
staging, and kernel work are co-dominant; no batching level in this
representation separates them by more than the noise band. A register-held
kernel variant would attack only the ~1.5–2 ns of arithmetic left per
source — bounded under ~4% end-to-end, below the keep bar.

Both layers were reverted (the fgf branch deleted; gfm tree restored); the
only surviving artifact is this record. The back-substitution loop, like
the release loop before it, is at a representation-bound local optimum:
the next step change requires pivot supports in a form that removes the
per-entry enumeration itself.

### Rejected: full-width packed active supports and lane-major release (2026-08-27)

The remaining sparse-loop share suggested promoting binary active supports
from sorted `u32` lists to full-width `u64` rows. Temporary instrumentation
on the real consumer systems (`raptor-q` prepare and final 5%-loss decode at
`K = 56403`, `T = 64`) tested the premise before any production
representation was added. Both solves produced the same coefficient
schedule:

- 57,326 columns and rows;
- 379,929 binary merges touching 56,955 rows;
- every binary pivot source carried exactly one active entry;
- destination active weights never increased: min/p25/p50/p75/p90/p95/p99/max
  = 1/4/15/82/145/166/183/187;
- the current list merge walked 17,875,978 entries in total;
- destination/source frozen storage averaged 6.46/6.82 words per merge.

A full-width active row at this size is 896 words (7,168 bytes). Simulating
one-way promotion at thresholds 32, 64, or 128 promoted the same 907 rows
(6.50 MB) and sent 167,840 merges through the packed path. Counting one unit
per list entry or packed word, those paths perform 8.52x the current
iteration work. Threshold 16 promotes 4,036 rows (28.93 MB) and models at
12.36x. Threshold 256 or higher promotes nothing because the observed
maximum active weight is 187.

Standalone twins measured the exact hot operation: XOR a binary pivot whose
active support is the singleton pivot column. Pinned CPU, criterion `--quick`;
medians:

| Active weight | current sorted-list XOR | direct singleton removal | full 896-word XOR |
| ---: | ---: | ---: | ---: |
| 15 | 19.0 ns | 7.0 ns | 396.5 ns |
| 82 | 94.6 ns | 28.2 ns | 393.2 ns |
| 166 | 173.7 ns | 48.8 ns | 389.2 ns |
| 187 | 192.5 ns | 56.0 ns | 384.6 ns |

Full-width XOR does not beat the current list walk until between weights 256
and 512, outside the real distribution. Against direct singleton removal,
the crossover moves between 1,024 and 2,048. Width sweeps show the same
scaling: the packed/direct crossover lies between weights 16–32 at 1,000
columns, 64–128 at 5,000, and 256–512 at 20,000. The pure-LT regression
shapes remain below those width-dependent crossovers. Packing can win an
isolated small-width, wider-row operation, but that does not justify a
second row tier for the max-`K` problem this design targeted.

The same instrumentation corrected the release cost model. One real
max-`K` solve fired 56,796 pivots and walked 10,988,074 support entries;
15.94 average live lanes (maximum 16) expand that to 175,122,735 scalar
lane operations. Average/max pivot support was 193.47/312 entries. A
lane-major standalone loop won at 128 entries × 16 lanes but lost at
270 × 16; the real mixed distribution required an end-to-end check.

Three adjacent max-`K` consumer prepare pairs:

| Pair | entry-major | lane-major | change |
| ---: | ---: | ---: | ---: |
| 1 | 443.40 ms | 558.49 ms | +26.0% |
| 2 | 478.21 ms | 546.98 ms | +14.4% |
| 3 | 435.41 ms | 544.12 ms | +25.0% |

The maximum-of-three comparison is +16.8%. Lane-major release is rejected.

The simpler direct singleton-removal twin passed the full public Hybrid
differential suite and improved adjacent max-`K` prepare pairs by 4.0%,
11.4%, and 7.7% (maximum-of-three −10.7%). That result is independent of
full-width packing and needs its own generic-shape A/B before any production
change. All temporary counters, environment switches, and microbench code
were removed after this record. No promotion threshold or packed-row budget
is retained.

## Sixth round: word-domain release and batched back-substitution (2026-08-27)

The measurement gate that stopped the packed-sparse rewrite left a corrected
phase table and one unlanded winning candidate. This round starts from a
release-profile instruction profile of the *real* consumer solve rather than
the phase timers, and lands every candidate that beat the noise band.

Host: Intel Core Ultra 7 258V, Linux, rustc 1.98.0, backend
`v3_gfni_crypto`, `taskset -c 2`. Consumer numbers come from two single-shot
drivers built against `raptor-q` at `perf/gfm-consumer-rounds` (`K = 56403`,
`T = 64`, one source block): max-`K` encoder preparation, and a 5%-loss
decode driven to completion. Each driver reports the minimum of four
(prepare) or three (decode) in-process iterations, and base/new binaries are
interleaved round by round. Synthetic numbers are Criterion medians from
`rfc_scale` and `hybrid`, `gfm`'s own `fgf` rev pin (so they exclude the
`fgf` prerequisite below).

### The instruction profile that set the order

`perf record` over max-`K` prepare, self time, entering state:

| Symbol | Share |
| --- | ---: |
| `Row::for_each_entry` (release substitution closure) | 28.2% |
| `run_into` (probe loop, dense assembly, back-substitution outer) | 17.4% |
| `largest_component_edge` | 8.9% |
| `Row::axpy_xor_parts` | 8.8% |
| `Row::for_each_entry` (back-substitution closure) | 6.0% |
| `xor_avx2_impl` | 5.2% |
| `Matrix::two_live_rows` | 4.6% |
| libc `memcpy`/`memset`/`realloc` | 5.7% |

Instruction-level annotation of the release closure put 91% of its samples
in five instructions: a destination-length bounds check reloaded every
iteration, the `acc[slot][lane]` byte load/store pair, and the scalar GF
multiply. Temporary counters on one max-`K` prepare (removed afterwards)
fixed the shapes the candidates had to serve:

| Counter | Value |
| --- | ---: |
| column-index probes | 379,929 |
| probes that fired a merge | 379,929 |
| stale probes | 0 |
| pivots whose active list was not the pivot column alone | 0 |
| mean active-list length at merge | 46.05 |
| maximum active-list length | 187 |
| release support visits | 10,988,074 |
| back-substitution support entries | 10,988,074 |
| weight-two tie-break calls | 155 |
| mean edges per tie-break call | 8,273.2 |
| mean distinct endpoints per call | 10,585.8 |

### Accepted, in landing order

Each row is the interleaved minimum-of-four max-`K` prepare after the change,
against the state before it.

| Change | prepare | change |
| --- | ---: | ---: |
| entering state | 427.5 ms | — |
| word-domain release accumulation + frozen-ordinal slot table | 348.7 ms | −18.4% |
| fused singleton merge + amortized-reset tie-break | 314.4 ms | −9.8% |
| back-substitution reads the dense block directly | 291.9 ms | −7.2% |
| back-substitution sources batched through `mul_add_gather` | 282.7 ms | −3.2% |
| union-by-size tie-break, no size pass, early-exit edge scan | 261.9 ms | −7.4% |
| `prepare_work` hygiene (lane/mask memsets, bucket span, dead-row index) | 249.4 ms | −4.8% |
| dense-block row-offset table for substitution sources | 242.2 ms | −2.9% |

**Word-domain release accumulation.** A packed pivot row's coefficients are
all one, so `acc[slot][lane] += factor[lane] · value` is `acc[slot] +=
factors`. Staging factors over the full lane width with zeros on dead lanes
turns the inner loop into one fixed-width array addition the compiler keeps
in vector registers, with no bounds check and no scalar multiply. The
release closure left the profile entirely: 28.2% → 2.1%. The earlier
"per-entry lane-vector `mul_add`" reject differs in that it called a
dispatched kernel per entry; this one calls nothing.

**Fused singleton merge.** Every binary pivot source carried exactly the
pivot column, so the sorted-list merge collapses to one search, one shift,
and the frozen word XOR — and the destination's new active weight is its new
list length, so the `is_active` recount disappears with it. This is the
candidate the previous gate identified and deferred; the generic-shape A/B it
was waiting for is the synthetic table below.

**Back-substitution over the dense block.** A pivot row carries its pivot
column and frozen bits over inactivated columns only, and every inactivated
column is a row of the already-final dense block. Reading sources there
instead of back through the solution matrix removes the aliasing split
(`two_live_rows`, 4.6% of the entering profile) and resolves each source
through one flat offset table. Sources are then staged in groups of 64 and
folded by one `mul_add_gather` call per group, which holds the destination
row in registers across the group: `xor_avx2_impl` fell 11.7% → 1.3%.

**Tie-break.** `largest_component_edge` cleared three column-wide arrays and
swept the column space on every one of its 155 calls. It now sizes its
arrays once, resets only its own endpoints through a sentinel, carries
component sizes in the union (no endpoint sweep), and stops the edge scan at
the first edge attaining the maximum size. The partition and every component
size are invariant under union-by-size, so the chosen edge is unchanged.
8.9% → 7.9% of a 1.7x smaller total.

**`prepare_work` hygiene.** The `n`-wide lane store and mask are sized, not
cleared — every lane read is gated on the mask, and the release pass zeroes
the `g` inactive slots it reads unconditionally. `pivot_time` was written and
never read; removed. Weight queues span the maximum input weight instead of
the column count, growing on demand from `bucket_move`. Deferred rows are
left out of the column index: they never pivot and never merge, and at max
`K` they contributed one entry per dense HDPC coefficient in every column.

### Rejected in this round

| Candidate | Result |
| --- | --- |
| Interned dense node universe for the tie-break | 239.7 / 247.7 / 263.4 ms against 241.2 / 238.5 / 240.6 ms — flat to worse, rejected |
| Linear scan-and-compact instead of binary search in the singleton merge | 247.1 / 244.7 / 242.8 ms against 240.8 / 242.5 / 240.6 ms — consistently ~1–2% worse, rejected |
| Per-entry `add_assign` instead of the batched gather | 255.1 / 250.9 / 252.2 ms against 242.8 / 241.8 / 243.3 ms — the gather wins ~4%, kept |

The interned-universe reject reproduces the previous round's finding that
compacting the tie-break's node space does not pay; the win there is
entirely in not touching the column space at all.

### Synthetic shapes, Criterion medians

| Case | before | after | change |
| --- | ---: | ---: | ---: |
| `rfc_scale` lt_only 1000 | 252.91 µs | 181.71 µs | −28.2% |
| `rfc_scale` lt_only 5000 | 1.6019 ms | 1.2592 ms | −21.4% |
| `rfc_scale` lt_only 10000 | 4.0118 ms | 3.1930 ms | −20.4% |
| `rfc_scale` lt_only 20000 | 9.4353 ms | 7.6048 ms | −19.4% |
| `rfc_scale` lt_hdpc_eager 500 | 10.050 ms | 9.0609 ms | −9.8% |
| `rfc_scale` lt_hdpc_eager 1000 | 62.649 ms | 53.690 ms | −14.3% |
| `rfc_scale` lt_hdpc_eager 2000 | 433.12 ms | 386.47 ms | −10.8% |
| `rfc_scale` lt_hdpc_eager 4000 | 3.3569 s | 2.9879 s | −11.0% |
| `rfc_scale` lt_hdpc_deferred 500 | 395.58 µs | 285.02 µs | −27.9% |
| `rfc_scale` lt_hdpc_deferred 1000 | 1.0266 ms | 749.68 µs | −27.0% |
| `rfc_scale` lt_hdpc_deferred 2000 | 3.4796 ms | 2.6235 ms | −24.6% |
| `rfc_scale` lt_hdpc_deferred 4000 | 13.101 ms | 9.8897 ms | −24.5% |
| `rfc_scale` lt_hdpc_deferred 56403 | 172.93 ms | 112.35 ms | −35.0% |
| `hybrid` k1000 hybrid | 434.43 µs | 362.67 µs | −16.5% |
| `hybrid` k1000 dense_ple (control) | 7.8008 ms | 7.7455 ms | 0.99x |

No shape regresses, the pure-LT family included, and the dense-`Ple` control
holds. The eager dense-band family — the one the first round accepted a 12%
regression on — improves 9.8–14.3%.

### Consumer, paired and interleaved

| Case | before | after | change |
| --- | ---: | ---: | ---: |
| `raptor-q` prepare K=56403 | 438.5 / 445.5 / 450.3 ms | 240.2 / 242.9 / 244.5 ms | −44.2% |
| `raptor-q` decode K=56403, 5% loss | 423.9 / 424.4 / 433.3 ms | 224.5 / 233.2 / 228.4 ms | −45.0% |

Percentages compare the worst measurement on each side. Every `gfm` and
`raptor-q` test suite is green, including the eager/deferred byte-identity
differentials, the dense-`Ple` oracle suites, the `with_initial_inactive`
suites, the inconsistency-rejection tests, `raptor-q`'s frozen fixtures and
`cberner/raptorq` interop vectors, and the zero-allocation steady-state
proof.

### Reference comparison, and what the reference was actually measuring

`cberner/raptorq` 2.0.1 (pin `83cf194`), same host, same drivers,
`taskset -c 2`. **`raptorq` carries a process-global, 64-entry
`SourceBlockEncodingPlan` cache keyed by symbol count**
(`src/encoder.rs:207`), so a second encode at the same `K` in the same
process replays a cached operation list instead of solving: 124 ms on the
first `SourceBlockEncoder::new`, 11 ms on every one after it. Only the
first-solve number is comparable to a `gfm` solve, so the reference is run
one iteration per fresh process.

| Case | reference | `raptor-q` before | gap | `raptor-q` after | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| prepare K=56403 | 129.6–131.2 ms | 438.5–450.3 ms | 3.39x | 240.2–244.5 ms | 1.86x |
| decode K=56403 | 121.0–123.5 ms | 423.9–433.3 ms | 3.48x | 223.6–226.5 ms | 1.84x |

The plan cache is also the measurement that sizes `raptor-q`'s own deferred
detached-schedule item: an 11x saving on repeated same-`K` encodes, entirely
codec-side.

### The `fgf` prerequisite, isolated

`gf8d`'s N-to-1 gather was wired to `gather_impl::<Affine8D, false, 4>`, so
for rows below the 128-byte main tile it fell through to one single-source
AXPY per source — the exact body `gather_gfni` stopped using for `Gf8B` when
the source-fused short-row rule landed ("Native GFNI source-fused short
rows", `fgf/BENCHMARKS.md`). Giving `gather_affine` the same selection rule
is worth, interleaved on the same drivers:

| Case | `gfm` only | `gfm` + `fgf` fused tail | change |
| --- | ---: | ---: | ---: |
| prepare K=56403 | 252.6 / 248.6 / 251.2 ms | 243.7 / 247.9 / 243.9 ms | −1.5…−3.5% |
| decode K=56403 | 237.7 / 234.6 / 233.7 ms | 232.8 / 232.5 / 229.4 ms | −1.8…−2.1% |

This lands in `fgf` first, on its own revision, before `gfm` repins — the
`gfm` numbers above are the pinned-`fgf` state and do not depend on it.

### What the profile says to do next

Entering state 427.5 ms, leaving state 242.2 ms. The leaving profile:

| Symbol | Share |
| --- | ---: |
| `run_into` (probe loop, dense assembly, back-substitution outer) | 28.0% |
| `Row::for_each_frozen_ordinal` (back-substitution staging) | 11.1% |
| `gather_impl::<Affine8D, true, 4>` | 9.0% |
| `Row::eliminate_singleton` | 7.9% |
| `Components::largest_component_edge` | 7.9% |
| libc `memcpy`/`memset`/`realloc` | 9.0% |

Back-substitution's remaining 20% is staging plus kernel, at ~4.5 ns for one
64-byte accumulate whose source is L1-resident. The kernel's floor is set by
the `VGF2P8AFFINEQB` latency in the accumulator's loop-carried chain, which
every coefficient being one makes unnecessary: an `fgf` XOR gather taking a
base, a row length, and an index list would remove both the staging and the
multiply chain. `Row::eliminate_singleton` is latency-bound on per-row heap
allocations (67% of its samples are the one binary search over a 46-entry
list in a separately allocated `Vec`), which is an arena question, not a
search question — the linear-scan reject above is the evidence that the
search itself is not the cost.

## Seventh round: the XOR-gather back-substitution (2026-08-27)

The sixth round's leaving profile put back-substitution's residual cost at
staging plus kernel — `for_each_frozen_ordinal` staging sources into fat
pointers (11.1%) and `gather_impl::<Affine8D>` multiplying by an implicit
one (9.0%), about 4.5 ns per 64-byte accumulate whose every coefficient is
one. This round moves that fold into `fgf` as a blocked XOR gather:
`ops::add_gather` folds byte-offset rows of the dense block with the
destination row held in AVX2 registers across the whole support, no
coefficient representation and no staged fat pointers at all. The packed
rows' ordinals stage into a reused `u32` scratch and the frozen-ordinal
offset table shrinks to `u32` (4 bytes per source instead of a 16-byte
slice). The group-of-64 batching and its `GATHER_GROUP` stack array are
gone: one call per pivot row.

### Kernel level

Interleaved minimum-of-twenty, one call, 64-byte rows, host
`v3_gfni_crypto`, `taskset -c 2`. The old side includes its staging cost,
as the production call site paid it:

| Region rows | Sources | staged all-ones gather | `add_gather` | change |
| ---: | ---: | ---: | ---: | ---: |
| 64 (L1) | 8 | 0.02 µs | 0.01 µs | 1.9–2.4x |
| 64 (L1) | 64 | 0.14 µs | 0.09 µs | 1.4–1.5x |
| 64 (L1) | 187 | 0.41–0.45 µs | 0.23–0.28 µs | 1.5–1.9x |
| 4096 (L2) | 64 | 0.14–0.16 µs | 0.09 µs | 1.5–1.7x |
| 4096 (L2) | 187 | 0.42–0.45 µs | 0.24 µs | 1.7–1.9x |

The microbenchmark and its numbers are recorded in `fgf`'s
BENCHMARKS.md, "Blocked XOR gather"; the harness was deleted after the
record.

### Consumer, paired and interleaved

Same drivers and protocol as the sixth round: `K = 56403`, `T = 64`,
minimum-of-four (prepare) / three (decode) per binary, base and new
binaries interleaved round by round, worst measurement on each side
compared.

| Case | base | new | change |
| --- | ---: | ---: | ---: |
| prepare K=56403 | 249.0 / 251.6 / 251.0 ms | 230.7 / 237.7 / 237.1 ms | −5.5% |
| decode K=56403, 5% loss | 232.6 / 236.9 / 235.4 ms | 222.5 / 224.0 / 223.5 ms | −5.4% |

The whole session's absolute numbers ran 3–6% above the sixth round's
recorded state (the base binaries measured 249–252 ms against the recorded
240–244 ms); the drift is environmental and cancels in the paired
comparison.

### Synthetic recheck, same session, same drift

Criterion medians on this host, compared against the sixth round's
recorded numbers taken hours earlier on the same machine:

| Case | sixth round | now | drift |
| --- | ---: | ---: | ---: |
| `rfc_scale` lt_hdpc_deferred/56403 | 112.35 ms | 114.51 ms | +1.9% |
| `rfc_scale` lt_hdpc_deferred/4000 | 9.8897 ms | 10.497 ms | +6.2% |
| `rfc_scale` lt_hdpc_deferred/1000 | 749.68 µs | 785.13 µs | +4.7% |
| `hybrid` k1000 hybrid | 362.67 µs | 387.74 µs | +6.9% |
| `hybrid` k1000 dense_ple (control) | 7.7455 ms | 8.1670 ms | +5.4% |

Every case, including the untouched dense-`Ple` control, drifts up within
the same band; the hybrid/control ratio is unchanged (4.68% → 4.75%). The
synthetics are parity within drift; the paired consumer numbers are the
evidence.

### The dense-block floor, updated

Back-substitution now spends its time in the gather itself: one unaligned
load and one XOR per source per lane, destination in registers. The
remaining visible costs in a release profile of this state are the dense
solve and the ordinary sparse loop. The next candidate this opens — an
`eliminate_singleton` arena so packed rows stop being individually
allocated `Vec`s — is unchanged from the sixth round's analysis and is a
`gfm`-internal question.

All suites green: `gfm`'s full matrix (default, all-features,
no-default-features), the zero-allocation steady-state proof,
`fgf`'s differential suite on the host and `SIMD_BACKEND=scalar` tiers,
`raptor-q`'s frozen fixtures and interop vectors.
