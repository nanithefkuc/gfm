# AGENTS.md

Working rules for `gfm`. Rustdoc owns API contracts, `README.md` owns adoption
and usage, `BENCHMARKS.md` owns public measurements, and `CHANGELOG.md` owns
release changes and migration guidance.

## Scope and ownership

`gfm` owns matrices over GF(2) and GF(2^m), rank-revealing decompositions,
streaming echelon state, and hybrid sparse/dense solving. It receives equations
and returns matrix facts or solutions; it is not a codec.

Field arithmetic and packed row operations belong to `fgf`. Structured
Cauchy and Vandermonde matrices belong to `structmat`; polynomial-matrix
reduction belongs to `polymat`. Wire formats, shards, graph generation,
rate adaptation, and recovery protocols belong to consumers.

## Required workflow

Run routine commands from the crate root through `just`:

```sh
just doctor
just test [ARGS]
just features
just test-tiers
just lint
just doc
just msrv
just validate
```

`just validate` is the pull-request gate. It includes the dependency check,
feature tests, backend sweep, unsafe check, and coverage gate. A focused
regression runs before the complete gate. The MSRV is Rust 1.93, edition 2024,
matching the field dependency's feature floor.

`justfile` is a byte-identical shared file; never edit the vendored copy.
Crate-specific values and recipes belong in `crate.just`.

## Algebra and storage contracts

- Batch rank, determinant, RREF, kernel, solve, and inverse derive from one
  `Ple` factorization per storage domain. Do not add another batch pivoting
  engine for a solved problem.
- `SmallMatrix<F, K>` is the bounded compact-matrix exception. Its public
  comparison against `Ple` belongs in `BENCHMARKS.md`; extending its order
  bound or adding another elimination requires independent correctness and
  paired measurements.
- Rank deficiency is a normal factorization result. Inconsistent systems and
  singular inverse requests use the documented `SolveError` variants.
- `fgf` defines `inv(0) == 0`; test pivots explicitly rather than inferring
  singularity from division.
- Dense storage has a 32-byte-aligned base, pitch divisible by 32, and zero
  padding. Bit storage follows `fgf::bits`'s LSB-first layout with pitch
  divisible by 64. These are type invariants, not caller cleanup duties.
- Logical row swaps update the row map. Physical compaction remains explicit.
- Reusable factorization and mutable execution scratch remain distinct.
  State which calls allocate, which warm workspace, and which reuse it.
- Use the ecosystem mutation vocabulary: `_into` overwrites, `_assign` reads
  and replaces its destination, `_add` accumulates, and `_with` uses prepared
  state. Output-writing free functions take the destination first.

## Modules and features

The existing `dense`, `bits`, `incremental`, and `hybrid` subtrees use `mod.rs`.
Keep that style. Parent modules hold documentation, declarations, re-exports,
and genuinely shared items; concrete implementations belong to their owners.

`src/internals.rs` is the sole feature-gated, re-export-only facade. It contains
no implementation bodies. The extension traits and implementations in private
`src/internal_api.rs` compile in every feature configuration and expose
otherwise crate-private inspection and experiment methods without making them
stable inherent methods. Combining these files under the facade's gate would
make `internals` control compilation rather than reachability.

`internals = []` activates no dependencies and carries no compatibility promise.
Benchmarks and integration targets using it declare `required-features`;
private unit tests need no facade. Never enable another crate's `internals` in
a runtime dependency.

The runtime dependency set is exact-pinned `fgf` and optional `rayon`.
`simd` implies `std`; `parallel` implies `std` and is off by default.
Without default features the crate uses `no_std` plus `alloc`; owning matrix
storage still allocates. Do not introduce `std` outside its functional gate.

## Backend selection

`gfm` owns no SIMD kernels. `backend_for::<F>()` delegates to `fgf`'s per-field
selection. `simdispatch`, reached through `fgf`, owns CPU detection, ordering,
and the process-startup downgrade-only `SIMD_BACKEND` request. Do not add a
second detector, override, or cache.

`crate.just` declares `v3_gfni_crypto v3 v2 v1 scalar`. This is a requested x86
sweep, not proof those tiers executed: unsupported requests can fall back.
Record resolved field backends when collecting measurements. The fingerprint
test writes the requested override and resolved field backends directly to stderr,
including during captured tier-test runs. Other architecture runs must account
for their actual field backend rather than treating x86 requests as coverage.

## Safety and tests

Library code uses `#![forbid(unsafe_code)]`; architecture intrinsics stay in
`fgf`. The test-only `GlobalAlloc` adapter in `tests/zero_alloc.rs` is outside
that crate attribute. Its unsafe forwarding operations require per-item
allowances and SINCE–THUS proofs under the umbrella rules; do not treat the
library's attribute as evidence that the entire repository contains no unsafe.

| Item | Residue | Proof |
| --- | --- | --- |
| `tests/zero_alloc.rs`: `CountingAllocator`'s `GlobalAlloc` implementation | Test-only allocator callbacks | Unchanged pointers and layouts forward to `System`; fallible TLS access and wrapping counter arithmetic prevent bookkeeping unwinding. Each forwarding block states its caller-supplied obligations. |

`MIRI` is empty, so `just unsafe-check` reports a skip. `COV_IGNORE` is empty:
every library line counts toward the coverage gate.

- Use the built-in Rust harness and deterministic helpers from `tests/common`.
- Public contracts belong in integration tests; private state belongs in local
  unit tests. Cross-domain comparisons use an independent algebraic oracle.
- Tests assert values, failure variants, boundaries, and state preservation.
  Do not pin source text, Debug formatting, error wording, field copies, or
  forwarding. Panic tests match only a stable operation-specific fragment.
- Allocation promises require counting-allocator coverage. Warmup and output
  allocation occur before the counted execution interval.
- A green forced-tier run is not execution evidence. Inspect the backend used
  by the operation and report unsupported coverage explicitly.

## Benchmarks

Use CPU 3 on the Intel Core Ultra 7 258V and CPU 8 on the Core i7-12700K
available as SSH alias `suisei-cachy`. Verify actual process affinity. Do not
constrain a multiworker scaling experiment to one CPU; record worker placement
separately from the serial campaign.

```sh
FEC_GOLDEN_CORE=3 RAYON_NUM_THREADS=1 just bench-save hybrid
FEC_GOLDEN_CORE=3 RAYON_NUM_THREADS=1 just bench hybrid
```

`bench-save NAME` and `bench NAME` select a benchmark target, not a named
baseline. Criterion's saved baseline is `before`. The public targets are
`hybrid` and `rfc_scale`; `tuning` mixes public comparisons with internal
variants, and `parallel` exposes the row dispatcher for threshold work.
Do not publish an entire target's output without classifying its operations.

Record public API and competitor timings only in `BENCHMARKS.md`. Each new run
adds rows to the relevant table, with units in headers and `-` for unavailable
values. Put factual reproduction details and caveats beneath tables, not
result commentary. Internal timers, profiling shares, tuning twins, candidate
journals, and rejected internal experiments belong in git-ignored storage.

A new campaign reruns its baseline in the same session, interleaves variants,
keeps an unchanged control, and records CPU, OS, toolchain, dependency versions,
actual operation backend, affinity, flags, geometry, aggregation, and warmup.
Check correctness before measuring. Never infer a speedup from cross-session
numbers or change a threshold from reasoning alone.

The native competitor harness is a separate ignored package under
`external-bench/competitors/`. FLINT, M4RI, M4RIE, and FFLAS-FFPACK must not
enter the published crate's build or dependency graph. Missing native libraries
must be reported; a successful command is not proof every comparator ran.

## Documentation and release

Public items and modules need summary sentences and complete errors, panics,
layout, and ownership contracts. Use third person, present tense, and plain
words. Keep measurements out of comments and guides, linking the public record
instead. Public files never cite private planning material.

The package allowlist includes sources, tests, benchmarks, and user-facing
release documents. `AGENTS.md`, `BENCHMARKS.md`, the command surface, CI, and
external tools stay outside the package. The changelog keeps an `Unreleased`
section above release entries; breaking entries include migration guidance.

Umbrella patches are for local integration only. Release verification and
lockfile regeneration run outside the umbrella against published dependencies;
do not edit pins or the umbrella patch table to hide a resolution failure.
Further Cargo commands beneath the umbrella can rewrite the registry lock.

Commit subjects use `gfm: short verb phrase`. One crate per pull request,
validation before opening it, dependencies landed and green before consumer
pins, and no publication without explicit authorization.
