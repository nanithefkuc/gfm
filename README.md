> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

> [!WARNING]
> `gfm` makes no constant-time guarantee. Pivot selection and sparse scheduling
> depend on input values, and field arithmetic comes from variable-time `fgf`.
> Handling secret matrices, coefficients, or payloads requires a separate audit.

# gfm — Galois Field Math

`gfm` provides dense, bit-packed, and hybrid sparse/dense linear algebra over
GF(2) and GF(2^m). It supports batch factorization, streaming equations, and
sparse systems with dense residual blocks for the matrix layer underneath
erasure coding.

| Main property | What it provides |
| --- | --- |
| Rank-revealing factorization | Reusable PLE/PLUQ decompositions for rank, determinant, RREF, nullspace, solve, and inverse. |
| Two storage domains | Packed field-element matrices and bit-packed GF(2) matrices. |
| Streaming equations | An echelon accumulator reports innovative, dependent, and inconsistent equations. |
| Hybrid solving | Sparse binary and field-valued equations, initial inactive columns, and deferred dense rows. |
| Reusable workspace | Caller-owned decomposition and solve scratch, plus fixed-shape refactorization. |
| Runtime SIMD | Row operations use `fgf`'s selected field kernels. |
| Portable builds | `no_std` with `alloc`; no architecture-specific compiler flags required. |

`gfm` is a solver, not a codec. It owns matrix layouts and composes field
arithmetic from [`fgf`](https://github.com/nanithefkuc/fgf). Structured matrices
such as Cauchy and Vandermonde belong to
[`structmat`](https://github.com/nanithefkuc/structmat); polynomial-matrix
reduction belongs to [`polymat`](https://github.com/nanithefkuc/polymat).
Consumers own wire formats, shards, and recovery protocols.

## Installation

The minimum supported Rust version is 1.93, edition 2024. Examples use the
same `fgf` release as the library so field types match:

```toml
[dependencies]
gfm = "=1.0.0"
fgf = "=1.2.1"
```

For portable `no_std` execution, matrix storage still requires `alloc`:

```toml
[dependencies]
gfm = { version = "=1.0.0", default-features = false }
fgf = { version = "=1.2.1", default-features = false }
```

## Quick start

Factor a matrix once, then solve using reusable workspace. This system is over
GF(256) with the AES reduction polynomial; addition is XOR:

```rust
use fgf::Gf8B;
use gfm::{Matrix, Ple, PleScratch, SolveScratch};

// x + y = 3, y = 2, so x = 1.
let a = Matrix::<Gf8B>::from_rows(2, 2, &[1, 1, 0, 1]).unwrap();
let rhs = Matrix::<Gf8B>::from_rows(2, 1, &[3, 2]).unwrap();
let mut factor_scratch = PleScratch::new();
let factor = Ple::decompose(a, &mut factor_scratch);
assert_eq!(factor.rank(), 2);

let mut solution = Matrix::<Gf8B>::zeros(2, 1).unwrap();
let mut solve_scratch = SolveScratch::new();
factor.solve_into(&rhs, &mut solution, &mut solve_scratch).unwrap();
assert_eq!(solution.row(0), &[1]);
assert_eq!(solution.row(1), &[2]);
```

`Ple::decompose` consumes the input matrix. The factorization can solve further
right-hand sides without changing the coefficients. `Ple::redecompose_scratch`
reuses its storage for a new matrix of the same shape; its closure receives a
zeroed matrix to fill. Scratch and output storage are separate from the
reusable factorization.

## Matrices and decompositions

| Surface | Contract |
| --- | --- |
| `Matrix<F>` | Owning dense matrix of packed field elements, with borrowed `View` and `ViewMut` access. |
| `BitMatrix` | Owning GF(2) matrix with one element per bit. |
| `Perm` | Index-vector permutations with application, inverse application, composition, and parity. |
| `Ple<F>` / `bits::Ple` | Rank-revealing factorization in the dense or bit-packed storage domain. |
| `SmallMatrix<F, K>` | Compact square matrices within the documented order bound. |
| `solve_lower_unit_assign` / `solve_upper_assign` | Triangular solves in place, with the destination first. |
| `mul_into` / `mul_add` | Matrix multiplication into an output or accumulated into it. |

A decomposition provides `rank`, `det`, `rref_into`, `kernel_into`,
`solve_into`, `inverse_into`, and row and column rank profiles. Rank deficiency
is a normal result, not a decomposition error. Dense `solve_into` chooses the
solution with free variables zero; an inconsistent system returns
`SolveError::Inconsistent` without changing its output. Each operation documents
its required output shape and whether invalid geometry returns an error or
panics.

The [dense module](https://docs.rs/gfm/1.0.0/gfm/dense/index.html) defines the
field-element matrix API; the
[bits module](https://docs.rs/gfm/1.0.0/gfm/bits/index.html) provides the GF(2)
counterpart.

## Streaming and sparse systems

`Echelon<F>` absorbs equations one at a time and returns an `Innovation`
verdict: innovative, dependent, or inconsistent. Its reduced mode propagates
recovered unit rows so determined variables can be read during the stream;
forward-echelon mode retains equations for recoding.

`Hybrid<F>` accepts binary or field-valued sparse rows and reduces the residual
system through a dense factorization. Initial inactive columns and deferred
dense rows describe which equations participate in sparse scheduling.
`solve_into` re-analyzes the system; `replace_rhs` and `resolve_into` explicitly
reuse coefficient analysis for changed payloads.

The [incremental module](https://docs.rs/gfm/1.0.0/gfm/incremental/index.html)
and [hybrid module](https://docs.rs/gfm/1.0.0/gfm/hybrid/index.html) define input
validation, storage reuse, and solution access.

## Features

| Feature | Effect |
| --- | --- |
| default | enables `std` and runtime-dispatched `simd` |
| `std` | enables `fgf` runtime support |
| `simd` | enables `fgf` architecture kernels; implies `std` |
| `parallel` | optional Rayon symbol-axis row updates; implies `std` and is off by default |
| `internals` | re-export-only facade of implementation types and tuning APIs |

Without default features, the crate uses `no_std` plus `alloc` and portable
kernels. There is no separate `alloc` feature: owning matrices allocate.
Nothing behind `internals` is a compatibility promise; those APIs may change
or disappear in any release.

## Platforms and backends

`backend_for::<F>()` reports the backend used by `fgf` for a particular field.
Available field kernels include x86, `AArch64`, WebAssembly SIMD, and portable
fallbacks; not every field supports every backend. `gfm` owns no SIMD kernels
and adds no separate CPU detector.

[`simdispatch`](https://github.com/nanithefkuc/simdispatch) owns detection and
the downgrade-only `SIMD_BACKEND` override. It may request a weaker backend
at process startup; unsupported upgrades are ignored.

## Layout and safety

Dense rows use `fgf`'s little-endian field encodings. `Matrix<F>` owns a
32-byte-aligned base, a pitch that is a multiple of 32 bytes, and zero padding.
`BitMatrix` uses `fgf::bits`'s LSB-first bit encoding with a pitch that is a
multiple of 64 bytes. Logical row exchanges use a row map; `compact_rows`
restores contiguous physical row order.

The crate forbids unsafe Rust. Its dependencies supply the architecture-specific
row kernels. Safe memory access does not imply constant-time execution:
data-dependent pivots, scheduling, and field operations remain visible to
timing analysis.

## Performance

[BENCHMARKS.md](https://github.com/nanithefkuc/gfm/blob/main/BENCHMARKS.md)
records public-operation measurements and competitor comparisons. The public
benchmark targets are `hybrid` and `rfc_scale`; `tuning` and `parallel` also
exercise implementation details whose timings stay outside the public record:

```sh
FEC_GOLDEN_CORE=3 RAYON_NUM_THREADS=1 just bench-save hybrid
FEC_GOLDEN_CORE=3 RAYON_NUM_THREADS=1 just bench hybrid
```

The competitor harness links external libraries and is separate from the
published crate. The measurement record lists its reproduction requirements.

## Building

From the repository, `just` provides the build and verification commands:

```sh
just build       # release build with all features
just features    # no-default, default, and all-feature tests
just test
just doc
just validate    # complete CI gate, including coverage
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/gfm/blob/main/LICENSE).
