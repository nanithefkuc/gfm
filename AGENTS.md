# AGENTS.md

Working rules for `gfm`. Rustdoc owns API contracts, `README.md` owns adoption
and usage, `BENCHMARKS.md` owns public measurements, and `CHANGELOG.md` owns
release changes and migration guidance.

## Scope and ownership

`gfm` owns matrices and linear systems over 𝔽q: dense matrices over valid
`fgf::FieldBuffer` presentations and bit-packed GF(2) matrices, rank-revealing
PLUQ and its derived queries, incremental echelon under row insertion and
leading-column eviction, and sparse-to-dense inactivation solving. It receives
equations and returns matrix facts or solutions; it is not a codec.

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

The `dense`, `bits`, `incremental`, and `hybrid` subtrees use module-named
files (`src/dense.rs` beside `src/dense/`). Keep that style. Parent modules
hold documentation, declarations, re-exports, and genuinely shared items;
concrete implementations belong to their owners.

`src/internals.rs` is the sole feature-gated, re-export-only facade. It contains
no implementation bodies. The extension traits and implementations in private
`src/inspection.rs` compile in every feature configuration and expose
otherwise crate-private inspection and experiment methods without making them
stable inherent methods. The hybrid counter type is reachable only through the
same facade. Combining these files under the facade's gate would make
`internals` control compilation rather than reachability.

`internals = []` activates no dependencies and carries no compatibility promise.
Benchmarks and integration targets using it declare `required-features`;
private unit tests need no facade. Never enable another crate's `internals` in
a runtime dependency.

The runtime dependency set is exact-pinned `fgf` and optional `rayon`.
`simd` implies `std`; `parallel` implies `std` and is off by default.
Without default features the crate uses `no_std` plus `alloc`; owning matrix
storage still allocates. Do not introduce `std` outside its functional gate.

## Public API surface

`gfm` adopts umbrella A1–A10 in full. Public modules are facades: they hold
documentation, attributes, module declarations, and explicit re-exports, never
operation bodies. Each stable item has exactly one path.

The stable module set is closed for the major version: `dense`, `bits`,
`incremental`, `hybrid`, and the feature-gated `internals` facade. `error` is
private and surfaces only `GeometryError` and `SolveError` at the crate root.
No approval adds a stable module inside the major.

New stable items are placed by this ordered first-match list:

1. An owned value, borrowed view, scratch workspace, or prepared form over
   packed field elements goes in `dense`.
2. An owned value, borrowed view, scratch workspace, or prepared form over
   GF(2) bits goes in `bits`.
3. An operation over an incrementally maintained echelon, or its verdict or
   retained-row view, goes in `incremental`.
4. An operation over sparse equations or their dense reduction, or its
   solution, goes in `hybrid`.
5. An error type shared across storage domains stays at the crate root.
6. Otherwise, stop and ask the maintainer.

Prepared state, borrowed views, and their capability traits follow their
operation's module; backend representations stay private or behind
`internals`. The stable API exposes no backend queries: no tier list,
threshold, capability boolean, or kernel name appears in it. The committed
inventory under `api/` is the source-level contract, `just api` checks it, and
regenerating it happens only inside an authorized API change.

## Backend selection

`gfm` owns no SIMD kernels. Callers use `fgf::backend_for::<F>()`; the crate's
private blocking policy reads the same answer. `gfm`'s stable API exposes no
backend query or backend type. `simdispatch`, reached through `fgf`, owns CPU
detection, ordering, and the process-startup downgrade-only `SIMD_BACKEND`
request. Do not add a second detector, override, or cache.

`crate.just` declares `v4x v3_gfni_crypto v3 v2 scalar`, the x86-64 tiers
`fgf` 4.0.1 routes; `v4x` needs the `simd512` feature, which the all-feature
sweep enables. This is a requested sweep, not proof those tiers executed:
unsupported requests can fall back.
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
