# gfm

> `gfm` is a solver, not a codec. Field arithmetic and byte-buffer vector
> primitives come from `fgf` — never re-implement them here. Sparse graph
> topology, Tanner-graph generation, and peeling belong to `sgraph`. Wire
> formats, shard ownership, rate adaptation, degree distributions, and codec
> shells belong to consumers. This crate receives a matrix and returns facts
> about it.

## Non-negotiables

1. **One elimination per storage domain.** Every rank, determinant, inverse,
   echelon form, kernel, and solve is a reader of `Ple`. Adding a second
   pivoting loop is the defect this crate was created to remove.
2. **No `unsafe`.** Forbidden at the crate root. Vector kernels come from
   `fgf`.
3. **One dependency.** `fgf`, pinned by rev. `rayon` optional and off. Adding
   `sgraph`, `lattica`, `cafft`, `simdispatch`, or `archmage` inverts or
   crosses the stack's layering; CI fails the build.
4. **Never re-host field arithmetic.** Call `fgf::ops` directly. Do not wrap,
   rename, or re-export it under a new name.
5. **Layout is a type invariant.** 32-byte aligned base, pitch a multiple of
   32, padding zero and staying zero. Breaking one silently costs a measured
   1.4x.
6. **Rank deficiency is not an error.** Only inconsistency is.
7. **`inv(0) == 0` is inherited.** Test pivots with `is_zero()`; never infer
   singularity from a division result.
8. **Numbers live in `BENCHMARKS.md`.** Doc comments state the decision and
   the mechanism and point there.
9. **Do not land a performance change on reasoning alone.** A/B it, keep both
   twins compiled, record the ratio.
10. **Oracles stay independent.** An implementation is never its own test.

## Working here

- Edition 2024, MSRV 1.89. No toolchain pin; select `+1.89.0` explicitly for
  the MSRV check.
- Features: `default = ["std", "simd"]`; `simd` implies `std`; `parallel` is
  an off-by-default no-op placeholder; `internals` exposes this crate's
  unstable surface (never `fgf`'s — we do not enable it).
- `src/lib.rs` and every `mod.rs` hold declarations only — no function
  bodies, no `impl` blocks. Public items are re-exported at the crate root.
- Errors are hand-rolled in `src/error.rs`: small enums per failure domain,
  struct variants carrying the offending value and the limit, manual
  `Display`, `std::error::Error` under `std`. Every fallible public function
  documents `# Errors`.
- Test placement follows visibility: in-module `#[cfg(test)]` for private
  state, `tests/` for the public surface. Fixed-seed LCG only (`fgf`'s
  `noise(len, seed)` shape); no `rand`, no nondeterminism. Exact values, not
  predicates.
- The full check set:

  ```sh
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  cargo clippy --all-targets --no-default-features -- -D warnings
  cargo test
  cargo test --all-features
  cargo test --no-default-features
  RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
  cargo build --target aarch64-unknown-linux-gnu --no-default-features
  cargo build --target wasm32-unknown-unknown --no-default-features
  cargo +1.89.0 build --all-features
  ```

- Benchmarks go through `criterion`; baselines are deliberately not
  committed. Measurement hygiene: interleave base/new, take the maximum of at
  least three runs, keep an unchanged 1.00x control, treat 16–128 byte rows
  as noise.
- Commit subjects are at most ~10 words, shaped `gfm: short verb phrase`.
  What changed and why lives in the pull request and `CHANGELOG.md`.
