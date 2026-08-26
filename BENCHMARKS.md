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
