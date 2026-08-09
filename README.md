> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# gfm — Galois Field Math

`gfm` is linear algebra over GF(2) and GF(2^m): a solver, not a codec. It
owns the rank-revealing PLE decomposition and everything derived from it —
rank, determinant, inverse, RREF, nullspace, solve — plus the incremental
echelon accumulator streaming codes absorb packets into, closed-form inverses
for the structured matrices FEC actually uses (Cauchy, Vandermonde), and a
hybrid sparse→dense solve that turns an O(n³) residual solve into an O(g³)
one on the inactivated block.

Field arithmetic and byte-buffer vector kernels come from
[`fgf`](https://github.com/nanithefkuc/fgf) and are never re-implemented
here. The crate is `#![forbid(unsafe_code)]`.

**Early development.** The public surface so far is the containers —
`Matrix<F>` (32-byte aligned, 32-byte pitch, zero padding) with its `View` /
`ViewMut` borrow API, `BitMatrix` (`u64` words, 64-byte pitch), and `Perm`
(LAPACK-style index-vector permutations) — and the decomposition in both
storage domains: `Ple<F>` over GF(2^m) and `bits::Ple` over GF(2) each
compute a rank-revealing `A = P·L·U·Q` once, and `rank`, `det`, `rref`,
`kernel_into`, `solve_into`, `inverse_into`, and both rank profiles are
readers of it. The two domains share the contract and the pivot order but no
elimination code: the GF(2) inner loop is a word-level masked XOR. On top of
the decompositions sit the streaming `Echelon<F>` accumulator (`absorb` →
`Innovation`, decoder or recoder via one flag) and the structured matrices
`Cauchy<F>` and `Vandermonde<F>` with their closed-form `O(k²)` inverses, plus
a generic `batch_invert`. `Hybrid<F>` accepts binary or field-valued sparse
rows, peels and permanently inactivates them into a small dense block, defers
symbol row operations until they are needed, and back-substitutes the result.
Its `solve_into` form reuses all workspaces and allocates nothing after warm-up.
Triangular solves (`solve_lower_unit_into`, `solve_upper_into`) and matrix
multiply (`mul_into`, `mul_add_into`) are exposed as the building blocks they
are. Nothing here is stable.

## Usage

The MSRV is Rust 1.89.

`gfm` is distributed through git only; it is not published to
[crates.io](https://crates.io), and the dependency on `fgf` is pinned to an
exact revision — a floating dependency is a format-break risk when the
dependency feeds wire bytes.

```toml
[dependencies]
gfm = { git = "https://github.com/nanithefkuc/gfm" }
```

Portable `no_std` builds are also available:

```toml
[dependencies]
gfm = { git = "https://github.com/nanithefkuc/gfm", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | runtime CPU detection and `fgf`'s vector kernels |
| `std` without `simd` | portable kernels |
| `parallel` | optional Rayon symbol-axis row updates for contiguous work at or above the measured 2 MiB threshold; off by default |
| `internals` | exposes unstable implementation types and tuning twins; not a compatibility promise |
| `--no-default-features` | `no_std`, portable kernels |

Backend selection is single-source: `Backend` is `simdispatch`'s ladder
re-exported through `fgf`, and the downgrade-only `SIMD_BACKEND` environment
variable (`v3_gfni_crypto`, `v3`, `v2`, `neon_aes`, `neon`, `wasm128`,
`scalar`) applies to the whole stack. `gfm` adds no override of its own.

## Building

```sh
cargo build --all-features
cargo test --all-features
cargo test --no-default-features
```

Cross-compiles to `aarch64-unknown-linux-gnu` and `wasm32-wasip1`.

## License

MIT — see [LICENSE](LICENSE).
