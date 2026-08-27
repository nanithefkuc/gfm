# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Deferred-row release runs in the inactive column space: substitutions
  accumulate into a `g`-wide accumulator indexed by inactive ordinal
  instead of an `n`-wide lane store, factors read straight from the seeded
  coefficients (substitutions never touch pivot columns), live lanes are
  hoisted out of the per-entry loop, and groups widen from sixteen to
  sixty-four lanes. Consumer prepare at max K drops another 19%; answers
  stay byte-identical under the eager/deferred differentials. See
  `BENCHMARKS.md`.
- Dependent-row verification in `Hybrid` runs in reduced form: released
  deferred rows evaluate their substituted inactive-column coefficients
  against their substituted right-hand side instead of re-walking L-wide
  input supports, and binary rows evaluate as a plain XOR of value rows.
  Per-row verdicts are equivalent identities, so inconsistency reports name
  the same row; deferral semantics and op counters are unchanged. Synthetic
  max-K solves drop ~31% and mid-range shapes up to ~43%. See
  `BENCHMARKS.md`.
- Packed frozen rows in `Hybrid`: each binary sparse row splits its support
  into the columns still active in the schedule and bit-packed words over
  the inactivated columns, keyed by a global frozen ordinal; freezing moves
  entries out of the lists once at inactivation time, so later combinations
  XOR whole words. Field-valued rows keep flat lists and widen packed rows
  exactly once on first contact. Answers, schedules, and stats are
  byte-identical; consumer prepare at max K drops ~60% and decode up to
  ~51%, with a measured +4% standing cost on one synthetic max-K shape.
  See `BENCHMARKS.md`.
- GF(2) trailing updates read a panel's L-factor selector in one masked
  byte-window load (`BitMatrix::row_selector`) rather than a bounds-checked
  bit read per pivot column, and walk set bits with `trailing_zeros`.
  Combined with `fgf::bits::RangeXor` preparing its XOR backend, the plain
  path drops 23–31% at order ≥128 and the remaining M4RI cutover penalty
  closes to 0.4–3.2%. Both changes are byte-for-byte identical and move the
  M4RI table crossover from 128 to 224. See `BENCHMARKS.md`.

### Added
- Coefficient-free binary rows in `Hybrid`: binary rows carry implicit
  unit coefficients and widen on first field contact; GF(2) merges
  degenerate to a support XOR. Cached weight-two edges, weight counting
  folded into the merge walk, inactivation decrements through the column
  index, and a batched deferred-row release (lane groups of sixteen, one
  pivot-time-ordered pass per group) complete the second scheduling
  round. A max-`K` RaptorQ solve drops from ~655 ms to ~278 ms in `gfm`
  and encoder preparation from ~1.9 s to ~1.36 s in the consumer. See
  `BENCHMARKS.md` for the measured strategy ladder.
- `Hybrid::push_deferred_field_row`: dense equations excluded from
  sparse-phase scheduling and released into the dense phase with their
  pivoted-column entries substituted out in one pivot-time-ordered pass.
  Deferral changes the schedule, never the answer — rank, solution, and
  inconsistency verdicts stay identical to the eager push. `SolveStats`
  gains `deferred_rows`.
- Indexed sparse-phase scheduling: incremental active-weight maintenance
  (no per-iteration recount), weight-bucketed minimum selection, and a
  column-to-row index so a pivot's elimination visits only the rows that
  contain its column. RFC-shaped solves improve from quadratic to
  near-linear; a max-`K` RaptorQ intermediate-symbol solve drops from
  ~300 s to ~0.7 s in `gfm` and ~2 s end to end in the consumer. See
  `BENCHMARKS.md` for the strategy-by-strategy measurements.
- `Hybrid::with_initial_inactive`: solver input that places a validated,
  sorted, distinct column set into the inactive set before sparse-phase
  scheduling, persisting across repeated solves. Permanently-inactive (PI)
  constructions pass their pre-inactivated columns here instead of relying
  on dynamic inactivation to find them. `SolveStats` gains
  `initial_inactivations` separating those from dynamic ones.
- Incremental leading-position tracking in `weak_popov_basis_with_scratch`:
  after each row reduction the schedule updates only the changed row's
  leading-term slot instead of re-scanning every row, falling back to a
  partial rebuild only when the target lands on a free column.
- Reusable `WeakPopovScratch`, row-based `weak_popov_with_scratch`, and indexed
  `weak_popov_basis_with_scratch` storage for allocation-free repeated
  polynomial-module reduction, including slab-backed consumer layouts.
- `Ple::redecompose_with` reuses a fixed-shape decomposition's matrix,
  permutation, and rank-profile storage for allocation-free repeated
  decompositions. The replacement matrix is supplied through a zeroed
  closure-bound view, preventing stale factors from entering the next solve.

- The streaming accumulator. `Echelon<F>` absorbs one equation at a time,
  returning an `Innovation` verdict (`Innovative { pivot }` / `Dependent` /
  `Inconsistent`), maintaining rank and — with its `reduced` flag —
  substituting recovered unit rows and propagating new units backward so
  decoded variables are readable as soon as they are determined. The flag
  collapses a decoder and a recoder (forward echelon only) into one type;
  `absorb` reuses internal scratch, so a steady-state stream allocates nothing.
  It agrees with a batch `Ple` over the same rows on rank, pivot columns, and
  solution, and is order-independent.
- Structured matrices. `Cauchy<F>` (`C[i][j] = (x_i + y_j)⁻¹`) with the
  contiguous (`indexed`), geometric-progression (`geometric`), and arbitrary
  (`from_points`) index-set policies, an `is_mds` exhaustive-minor check, and a
  closed-form `O(k²)` inverse that agrees entry-for-entry with `Ple::inverse`.
  `Vandermonde<F>` with `O(n²)` Lagrange inversion and a `# Warning` that its
  submatrices need not be nonsingular (a concrete singular submatrix is
  exhibited in the tests). A single generic `batch_invert` (Montgomery's
  trick) that agrees with elementwise `inv()` on every field, zero included.
- The hybrid sparse→dense solver. `Hybrid<F>` stores sorted sparse rows in
  binary form until a field-valued operation requires widening, applies the
  largest-component tie-break for weight-two rows, permanently inactivates
  columns into a dense `Ple` block, defers payload operations away from
  redundant rows, and back-substitutes the recovered values. Generated sparse,
  LDPC-shaped, stopping-set-shaped, and mixed-field systems agree with a full
  dense `Ple` on rank, unique solutions, and inconsistency. Deferred and eager
  application are byte-identical while deferred performs fewer counted payload
  row operations; `solve_into` is allocation-free after warm-up. On the
  recorded RFC Table 1 LT-degree-shaped `k = 1000` fixture, every measured run
  favored `Hybrid`, with a conservative paired-run speedup of 1.12x.
- Compact dense blocks. `SmallMatrix<F, K>` stores and factors square
  matrices of order at most 64 without pitch, a row map, or heap allocation.
  Its solve agrees with `Ple` at every order, and the one-byte, full-rank
  hybrid residual path now uses it through order 64; three pinned benchmark
  runs favored it at every supported order.
- Measured elimination tuning: shape/backend panel-width dispatch, an
  eight-pivot M4RI table path for GF(2) matrices from order 224, and retained
  Newton–John and unblocked twins behind `internals`. Every production result
  remains byte-identical to its untuned twin.
- Optional Rayon symbol-axis parallelism for contiguous field-row updates.
  It is off by default and begins at the measured 2 MiB work threshold; pivot
  selection and hybrid sparse scheduling remain serial.
- Reproducible same-host rank benchmarks against FLINT, M4RI, M4RIE, and
  FFLAS-FFPACK, with build-time discovery and loud runtime skips when a
  comparator is unavailable.

- The elimination. `Ple<F>` computes a rank-revealing `A = P·L·U·Q`,
  right-looking blocked with a rank-deficiency-safe panel factorization, and
  every derived query is a reader of it: `rank`, `det`, `rref_into`,
  `kernel_into`, `solve_into` (with a reusable `SolveScratch`), and
  `inverse_into`, plus both rank profiles. The decomposition agrees with the
  unblocked reference byte for byte on `lu`, `p`, `q`, and both profiles,
  independent of panel width; differential coverage against FLINT's
  `fq_nmod_mat_lu` runs where the library is present.
- The GF(2) elimination. `bits::Ple` computes the same rank-revealing
  `A = P·L·U·Q` over `BitMatrix`, with a separate word-level inner loop:
  pivot search is `trailing_zeros` on the column bitmap and column
  elimination is a masked `u64` XOR with no coefficient and no
  normalization. Its derived queries mirror the dense set — `rank`, `det`
  (a `bool`), `rref_into`, `kernel_into`, `solve_into` (with a reusable
  `SolveScratch`), `inverse_into`, and both rank profiles. The same logical
  matrix carried through `bits::Ple` and through `dense::Ple<Gf8>`
  one-bit-per-byte yields identical rank, RREF, rank profiles, and kernel
  bases. Differential coverage, gated on the libraries being present, runs
  against M4RI's `mzd_echelonize` and FFLAS-FFPACK's `Rank` over GF(2), and
  against M4RIE's `mzed_ple` over GF(2^8) under fgf's `0x11B` field.
- Building blocks: triangular solves (`solve_lower_unit_into`,
  `solve_upper_into`) and matrix multiply (`mul_into`, `mul_add_into`),
  composed from `fgf::ops` row kernels.
- Containers. `Matrix<F>` over GF(2^m) with the layout invariants (32-byte
  aligned base, pitch a multiple of 32, padding zero and staying zero),
  `View` / `ViewMut` borrows with `split_rows` / `row` / `row_mut` /
  `swap_rows`, `BitMatrix` over GF(2) with `u64`-packed rows and a 64-byte
  pitch, and `Perm`, a LAPACK-style index-vector permutation with `apply` /
  `apply_inv` / `compose` / `parity`. Row exchange is an index operation on
  a row map; data moves only through `compact_rows`. Constructors validate
  geometry with checked arithmetic and return `GeometryError`.

### Changed

- `Hybrid`'s deferred-row release accumulates in the word domain. A packed
  pivot row's coefficients are all one, so its substitution is a
  whole-vector addition of the staged factor lanes per support entry
  instead of one scalar field multiply per live lane; dead lanes are staged
  as zero so the lane loop needs no live-lane list, and frozen ordinals
  resolve to their accumulator slot through a flat table. Answers and
  schedules are unchanged.
- The sparse phase eliminates a pivot column whose source row is that
  column alone through a fused path: one search, one shift, and the frozen
  word XOR, with the destination's new active weight read off its list
  length instead of recounted. This is the shape of every pivot in a
  peeling schedule; the general sorted merge stays for widened
  destinations and non-unit factors.
- Back-substitution reads its sources from the dense block rather than back
  through the solution matrix. A pivot row carries its pivot column and
  inactivated columns only, so every source is an already-final dense-block
  row reachable through one offset table, with no aliasing split; sources
  are folded in groups of sixty-four by one gather call each, which holds
  the destination row in registers across the group.
- The weight-two tie-break sizes its union-find once and touches only its
  own edge endpoints: component sizes ride along in the union, the endpoint
  sweep is gone, and the edge scan stops at the first edge attaining the
  maximum component size. The partition, every component size, and the
  chosen edge are unchanged.
- `Hybrid`'s per-solve setup stops clearing what it does not read: the lane
  store and mask are sized rather than zeroed (the release pass zeroes the
  slots it reads unconditionally), weight queues span the maximum input
  weight instead of the column count and grow on demand, deferred rows are
  left out of the column index, and the write-only `pivot_time` array is
  removed.

  Together these drop consumer preparation at max K by 44% and 5%-loss
  decode by 45%, improve every synthetic shape by 9.8–35% with the
  dense-`Ple` control flat, and close the `cberner/raptorq` gap at max K

- The GF(2) storage domain now delegates to fgf's native GF(2) surface
  instead of hand-rolling its own word loops. `BitMatrix` rows are packed
  in `fgf::bits` layout (one element per bit, LSB-first within each byte),
  and every row-vector operation — whole and masked-range XOR, prefix
  clearing, zero tests — goes through `fgf::bits` kernels; the crate keeps
  only geometry, indexing, and the pivot schedule. Public API follows the
  representation: `from_rows` takes packed bytes, `row` returns `&[u8]`,
  and `row_bytes` replaces `row_words`. `fgf` is pinned to `0.7.0`, whose
  GF(2^8) split renames the AES field `Gf8` → `Gf8B` (`gf8` → `gf8b`).
  A second round routes the hot applies through fgf's prepared-range
  split — `bits::RangeXor` + `xor_range_with`, prepared once per pivot or
  panel — cutting the measured elimination penalty from +22–46% to
  +5–7% at production sizes; see `BENCHMARKS.md`.

### Fixed
- Dense and bit-domain kernel, solve, and inverse results now undo multi-step
  column permutations in reverse swap order. The previous forward replay was
  only accidentally correct when the permutation was identity or involutive.
- The optional parallel row dispatcher now checks the 2 MiB threshold before
  asking Rayon for its thread count, so enabling `parallel` does not initialize
  the global pool or allocate on serial-sized steady-state operations.

## [0.0.0] - 2026-08-08

Initial scaffold. The crate exists so the containers and the decomposition
have somewhere to land; the only public surface is the error model
(`GeometryError`, `SolveError`, `ReduceError`) and the backend seam
(`backend_for`, re-exported `Backend`). One runtime dependency: `fgf`,
pinned by rev. Features: `std` + `simd` by default, an off-by-default no-op
`parallel` placeholder, and `internals` for this crate's own unstable
surface. Git-only; not published to crates.io.
