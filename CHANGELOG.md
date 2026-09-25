# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## Unreleased

## [1.0.1] - 2026-09-25

### Changed

- The `fgf` pin moves to `=1.2.1`, matching the rest of the published
  closure so one field copy resolves. No API change; results are unchanged.

## [1.0.0] - 2026-09-21

### Added

- `Hybrid::replace_rhs` and `Hybrid::resolve_into`: the schedule, the
  reduced coefficients, the pivot order, and the dense factorization are
  functions of the pushed equations' coefficients alone, so a system whose
  payloads changed pays the payload pass only. The answer, the
  determinedness, and the inconsistency verdict are the ones `solve_into`
  would give, proven by a differential test against a fresh solver. Reuse is
  opt-in — `solve_into` always re-analyzes, so a benchmark that re-solves one
  system keeps measuring solves. See `BENCHMARKS.md`.
- Coefficient-free binary rows in `Hybrid`: binary rows carry implicit unit
  coefficients and widen on first field contact; GF(2) merges degenerate to a
  support XOR. Cached weight-two edges, weight counting folded into the merge
  walk, inactivation decrements through the column index, and a batched
  deferred-row release complete the second scheduling round. See
  `BENCHMARKS.md`.
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
  near-linear; see `BENCHMARKS.md`.
- `Hybrid::with_initial_inactive`: solver input that places a validated,
  sorted, distinct column set into the inactive set before sparse-phase
  scheduling, persisting across repeated solves. Permanently-inactive (PI)
  constructions pass their pre-inactivated columns here instead of relying
  on dynamic inactivation to find them. `SolveStats` gains
  `initial_inactivations` separating those from dynamic ones.
- `Ple::redecompose_scratch` reuses a fixed-shape decomposition's matrix,
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
- The hybrid sparse→dense solver. `Hybrid<F>` stores sorted sparse rows in
  binary form until a field-valued operation requires widening, applies the
  largest-component tie-break for weight-two rows, permanently inactivates
  columns into a dense `Ple` block, defers payload operations away from
  redundant rows, and back-substitutes the recovered values. Generated sparse,
  LDPC-shaped, stopping-set-shaped, and mixed-field systems agree with a full
  dense `Ple` on rank, unique solutions, and inconsistency. Deferred and eager
  application are byte-identical while deferred performs fewer counted payload
  row operations; `solve_into` is allocation-free after warm-up. See
  `BENCHMARKS.md`.
- Compact dense blocks. `SmallMatrix<F, K>` stores and factors square
  matrices of order at most 64 without pitch, a row map, or heap allocation.
  Its solve agrees with `Ple` at every order, and the one-byte, full-rank
  hybrid residual path uses it through that bound. See `BENCHMARKS.md`.
- Measured elimination tuning: shape/backend panel-width dispatch, an
  eight-pivot M4RI table path for GF(2) matrices from a measured order, and
  retained Newton–John and unblocked twins behind `internals`. Every
  production result remains byte-identical to its untuned twin. See
  `BENCHMARKS.md`.
- Optional Rayon symbol-axis parallelism for contiguous field-row updates.
  It is off by default and begins at a measured work threshold; pivot
  selection and hybrid sparse scheduling remain serial. See `BENCHMARKS.md`.
- The elimination. `Ple<F>` computes a rank-revealing `A = P·L·U·Q`,
  right-looking blocked with a rank-deficiency-safe panel factorization, and
  every derived query is a reader of it: `rank`, `det`, `rref_into`,
  `kernel_into`, `solve_into` (with a reusable `SolveScratch`), and
  `inverse_into`, plus both rank profiles. The decomposition agrees with the
  unblocked reference byte for byte on `lu`, `p`, `q`, and both profiles,
  independent of panel width.
- The GF(2) elimination. `bits::Ple` computes the same rank-revealing
  `A = P·L·U·Q` over `BitMatrix`, with a separate word-level inner loop:
  pivot search is `trailing_zeros` on the column bitmap and column
  elimination is a masked `u64` XOR with no coefficient and no
  normalization. Its derived queries mirror the dense set — `rank`, `det`
  (a `bool`), `rref_into`, `kernel_into`, `solve_into` (with a reusable
  `SolveScratch`), `inverse_into`, and both rank profiles. The same logical
  matrix carried through `bits::Ple` and through `dense::Ple<Gf8>`
  one-bit-per-byte yields identical rank, RREF, rank profiles, and kernel
  bases.
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

- Replace historical benchmark tables with paired 258V/12700K public-operation
  and competitor results; consolidate environment and reproduction notes.
- **Breaking:** rename `solve_lower_unit_into(l, b)` to
  `solve_lower_unit_assign(b, l)` and `solve_upper_into(u, b)` to
  `solve_upper_assign(b, u)`. Both overwrite their right-hand-side operand.
  Rename matrix `mul_add_into` to `mul_add` and retained-row
  `mul_add_coefficients_into` to `mul_coefficients_add`; both accumulate.
- **Breaking:** rename `Echelon::has_pivot(column)` to
  `Echelon::is_pivot(column)`, querying the retained pivot state.
- Integration tests using the experimental facade declare their feature
  requirements in the manifest. Tier runs report resolved field backends.
- Correct singular-inverse and compact-matrix indexing documentation, document
  the test allocator's unsafe obligations, and remove implementation-pinning tests.

- **Breaking:** the minimum supported Rust version is 1.93, the floor `fgf`'s
  feature gating sets. To migrate, build with 1.93 or newer; no source change
  is required.
- **Breaking:** `fgf` is taken from its published release, pinned to an exact
  version, instead of a git revision. To migrate, a consumer that pinned
  `fgf` by git revision repoints it at the published version — a git pin and
  a registry pin resolve two copies of the field types, which do not
  interoperate.
- `gfm` targets its first stable registry release as `gfm = "=1.0.0"`.
  The package allowlist carries the crate sources, its tests and benches,
  and `README.md`, `LICENSE` and `CHANGELOG.md`; local working files and CI
  stay in the repository. Package keywords describe the finite-field matrix
  surface, and the README covers installation, solving, features, and safety.
  The lockfile resolves published dependencies without local path patches.
- A solve no longer pre-sizes every working row's frozen words to the
  whole inactivated-column space. A row carries a handful of frozen
  words; the reserve asked for one per inactivated column in every row,
  allocated and never read, and paid again by every per-block solver. See
  `BENCHMARKS.md`.
- The sparse phase stops allocating per row and per column at setup. The
  column-to-row index is one arena — a span per column in a flat buffer,
  laid out from the counted supports with slack, moved to a doubled span
  only when a merge overruns it, and reused across solves of the same
  equations — and packed rows share one merge scratch instead of carrying
  two buffers each, which also shrinks every row header the pivot probe
  touches. Freezing a column and widening a row compact in place. See
  `BENCHMARKS.md`.
- A field merge whose source adds no column to the destination runs in
  place, compacting over the entries that cancel, instead of building the
  result in a second buffer and swapping. This is every merge between two
  full-width rows. See `BENCHMARKS.md`.
- The dense solve reads the rank decomposition directly when the residual
  block has full row rank, which is when its independent subset is every
  row it holds. The second assembly and the second elimination of the same
  matrix are gone. See `BENCHMARKS.md`.
- Pivot inverses ride in the pivot list instead of being searched for
  again by the release substitution, back-substitution, and the kernel
  lift. Widened rows carry their pivot coefficient in an `n`-wide list, so
  the search was a full binary search per pivot per pass.
- Back-substitution folds its sources through `fgf`'s blocked XOR
  gather: the packed rows' frozen ordinals stage into a reused `u32`
  scratch, the frozen-ordinal offset table shrinks to `u32`, and one
  `ops::add_gather_offsets` call per pivot row replaces the staged
  fat-pointer groups and all-ones `mul_add_gather`. The group batching and
  its stack array are gone. See `BENCHMARKS.md`.
- `Hybrid`'s deferred-row release accumulates in the word domain. A packed
  pivot row's coefficients are all one, so its substitution is a
  whole-vector addition of the staged factor lanes per support entry
  instead of one scalar field multiply per live lane; dead lanes are staged
  as zero so the lane loop needs no live-lane list, and frozen ordinals
  resolve to their accumulator slot through a flat table. Answers and
  schedules are unchanged. See `BENCHMARKS.md`.
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
  are folded in groups by one gather call each, which holds the destination
  row in registers across the group.
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
  removed. See `BENCHMARKS.md`.
- Deferred-row release runs in the inactive column space: substitutions
  accumulate into a `g`-wide accumulator indexed by inactive ordinal
  instead of an `n`-wide lane store, factors read straight from the seeded
  coefficients (substitutions never touch pivot columns), live lanes are
  hoisted out of the per-entry loop, and lane groups widen. Answers stay
  byte-identical under the eager/deferred differentials. See
  `BENCHMARKS.md`.
- Dependent-row verification in `Hybrid` runs in reduced form: released
  deferred rows evaluate their substituted inactive-column coefficients
  against their substituted right-hand side instead of re-walking L-wide
  input supports, and binary rows evaluate as a plain XOR of value rows.
  Per-row verdicts are equivalent identities, so inconsistency reports name
  the same row; deferral semantics and op counters are unchanged. See
  `BENCHMARKS.md`.
- Packed frozen rows in `Hybrid`: each binary sparse row splits its support
  into the columns still active in the schedule and bit-packed words over
  the inactivated columns, keyed by a global frozen ordinal; freezing moves
  entries out of the lists once at inactivation time, so later combinations
  XOR whole words. Field-valued rows keep flat lists and widen packed rows
  exactly once on first contact. Answers, schedules, and stats are
  byte-identical. See `BENCHMARKS.md`.
- GF(2) trailing updates read a panel's L-factor selector in one masked
  byte-window load (`BitMatrix::row_selector`) rather than a bounds-checked
  bit read per pivot column, and walk set bits with `trailing_zeros`.
  Combined with `fgf::bits::XorRange` preparing its XOR backend, both
  changes are byte-for-byte identical and move the M4RI table crossover to
  a higher order. See `BENCHMARKS.md`.
- The GF(2) storage domain delegates to `fgf`'s native GF(2) surface
  instead of hand-rolling its own word loops. `BitMatrix` rows are packed
  in `fgf::bits` layout (one element per bit, LSB-first within each byte),
  and every row-vector operation — whole and masked-range XOR, prefix
  clearing, zero tests — goes through `fgf::bits` kernels; the crate keeps
  only geometry, indexing, and the pivot schedule. Public API follows the
  representation: `from_rows` takes packed bytes, `row` returns `&[u8]`,
  and `row_bytes` replaces `row_words`. `fgf`'s GF(2^8) split renames the
  AES field `Gf8` → `Gf8B` (`gf8` → `gf8b`). The hot applies route through
  `fgf`'s prepared-range split — `bits::XorRange` + `xor_range_with`,
  prepared once per pivot or panel. See `BENCHMARKS.md`.

### Removed

- **Breaking:** the structured-matrix surface — `Cauchy`, `Vandermonde`,
  `batch_invert`, `cauchy_inverse_coefficients_into`, `cauchy_scratch_len`,
  and the `GeometryError::Capacity`/`Collision` variants only it produced —
  moved to the `structmat` crate. To migrate, depend on `structmat` and
  import the items from its root, matching construction failures against its
  `StructureError`; `Matrix`, `Ple`, and every other `GeometryError` variant
  are unchanged.
- **Breaking:** the polynomial-module surface — `PopovLeadingTerm`,
  `WeakPopovRow`, `WeakPopovBasis`, `WeakPopovScratch`, `weak_popov`,
  `weak_popov_scratch`, `weak_popov_basis_scratch`, and the `ReduceError`
  only they produced — moved to the `polymat` crate. To migrate, depend on
  `polymat` and import the items from its root, matching reduction failures
  against its `ReduceError`; matrices over Fq stay here. Cancelling row
  updates in the new owner use the negative leading-coefficient ratio, and a
  collision between rows of equal leading degree reduces the lower-indexed
  row, so an odd-characteristic result is corrected and a tied basis may
  reduce to a different, equally valid weak Popov form.
- **Breaking:** `poly-ring` leaves the dependency set with the Vandermonde
  inverse that needed it. The runtime set is `fgf` and optional `rayon`. To
  migrate, a consumer that reached `poly-ring` types through this crate
  depends on `poly-ring` directly or on `structmat`, which owns that inverse.
- The competitor comparison harness — its benchmark target, build script,
  link shims, and foreign differential tests — leaves this repository and its
  package. M4RI and M4RIE are GPL and FLINT and FFLAS-FFPACK are LGPL, so the
  code that links them is a development tool held outside the published tree,
  in a crate-local, git-ignored `external-bench/competitors/` package.
  `BENCHMARKS.md` keeps the measurements it produced, with the caveat that
  reproducing them means writing that harness against the installed
  libraries.

### Fixed

- Dense and bit-domain kernel, solve, and inverse results now undo multi-step
  column permutations in reverse swap order. The previous forward replay was
  only accidentally correct when the permutation was identity or involutive.
- The optional parallel row dispatcher checks its work threshold before
  asking Rayon for its thread count, so enabling `parallel` does not
  initialize the global pool or allocate on serial-sized steady-state
  operations.

## [0.0.0] - 2026-08-08

Initial scaffold. The crate exists so the containers and the decomposition
have somewhere to land; the only public surface is the error model
(`GeometryError`, `SolveError`, `ReduceError`) and the backend seam
(`backend_for`, re-exported `Backend`). One runtime dependency: `fgf`,
pinned by rev. Features: `std` + `simd` by default, an off-by-default no-op
`parallel` placeholder, and `internals` for this crate's own unstable
surface. Git-only; not published to crates.io.
