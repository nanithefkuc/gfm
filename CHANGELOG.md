# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
  eight-pivot M4RI table path for GF(2) matrices from order 128, and retained
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
