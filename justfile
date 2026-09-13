# Shared FEC crate tooling. THIS FILE IS VENDORED VERBATIM into every crate
# repository; the canonical copy lives in the local `fec` umbrella at
# `tooling/justfile`. Every vendored copy is byte-identical, which is what makes
# drift a checksum comparison.
#
# Do not edit a crate's copy. Edit the canonical one and re-run `just sync` from
# the umbrella. Crate-specific values and recipes belong in `crate.just`, which
# this file imports, and are documented in that crate's `AGENTS.md`.

set shell := ["bash", "-uc"]

import 'crate.just'

# Runtime dependencies allowed outside the ecosystem (ground rule 3). Dev
# dependencies are not checked: the benchmark harness, property testing, and the
# competitor baselines of `BENCHMARKS.md` are dev-only by design.
EXTERNAL := "archmage rayon"

# Ecosystem crates, which may depend on each other subject to the layering in
# the umbrella README's dependency graph.
ECOSYSTEM := "simdispatch fgf butterfly-fft poly-ring polymat gfm structmat fec-graph softmetric lattica ratematch funcfield sgraph syndrome-engine gs-engine reliability-engine bp-engine lattice-engine lc-engine systematic-rs srs mix-dpc ccrlnc raptor-q latticode ldpc contort multiplicity reed-muller polar ag-codes"

# Minimum line coverage, enforced in CI (ground rule 6).
COV_MIN := "95"

_default:
    @just --list --unsorted

# ── Build ─────────────────────────────────────────────────────────────────────

[group('build')]
build:
    cargo build --release --all-features

[group('build')]
clean:
    cargo clean
    rm -f perf.data perf.data.old

# ── Correctness ───────────────────────────────────────────────────────────────

# Single-configuration run on the host's best backend.
[group('test')]
test *ARGS:
    cargo test --all-features {{ARGS}}

# Dispatch resolves one backend per process, so the plain run only ever
# exercises the host's best.
#
# Test once per declared backend tier.
[group('test')]
test-tiers:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test --all-features
    if [[ -z "{{TIERS}}" ]]; then
        echo "no dispatch surface in {{CRATE}}; host run only"
        exit 0
    fi
    for tier in {{TIERS}}; do
        echo "── SIMD_BACKEND=${tier}"
        SIMD_BACKEND="${tier}" cargo test --all-features
    done

# Feature closure: no-default (the no_std closure), default, everything.
[group('test')]
features:
    cargo test --no-default-features
    cargo test
    cargo test --all-features

[group('test')]
msrv:
    cargo +{{MSRV}} check --all-features --all-targets

# Crates that forbid unsafe have no `MIRI` args and skip; kernel owners list
# one `cargo miri test` argument set per line in `crate.just`.
#
# Miri over the scalar paths.
[group('test')]
unsafe-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{MIRI}}" ]]; then
        echo "{{CRATE}} declares no unsafe surface; nothing for miri to check"
        exit 0
    fi
    while IFS= read -r args; do
        [[ -z "${args}" ]] && continue
        echo "── cargo +nightly miri test ${args}"
        # shellcheck disable=SC2086
        MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test ${args}
    done <<< "{{MIRI}}"

# ── Lint and docs ─────────────────────────────────────────────────────────────

[group('lint')]
lint:
    cargo fmt --all --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings

[group('lint')]
fmt:
    cargo fmt --all

[group('lint')]
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

[group('lint')]
doc-open:
    RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --open

# Runtime dependency allowlist (ground rule 3). Dev dependencies are exempt.
[group('lint')]
deps:
    #!/usr/bin/env bash
    set -euo pipefail
    allowed=$(printf '%s\n' {{EXTERNAL}} {{ECOSYSTEM}} {{CRATE}} | sort -u)
    # Direct dependencies only: what an approved dependency pulls in below
    # itself is that crate's business, not this crate's rule-3 surface.
    # Stderr is dropped deliberately: patch-bookkeeping advisories (e.g. an
    # umbrella patch for a crate this graph does not use) are not dependency
    # names and must not pollute the allowlist comparison.
    tree=$(cargo tree -e normal --all-features --depth 1 --prefix none 2>/dev/null) || {
        echo "cargo tree could not resolve --all-features" >&2
        exit 1
    }
    found=$(echo "${tree}" | awk 'NF {print $1}' | sed 's/^[^a-zA-Z0-9_-]*//' | sort -u)
    extra=$(comm -23 <(echo "${found}") <(echo "${allowed}"))
    if [[ -n "${extra}" ]]; then
        echo "unapproved runtime dependencies (ground rule 3):" >&2
        echo "${extra}" >&2
        exit 1
    fi
    echo "runtime dependency set is within the allowlist"

# ── Coverage ──────────────────────────────────────────────────────────────────

# Line coverage: one run per backend tier, merged, then the 95% gate.
[group('cover')]
cover: _cover-runs
    #!/usr/bin/env bash
    set -euo pipefail
    ignore=({{ if COV_IGNORE == "" { "" } else { "--ignore-filename-regex " + COV_IGNORE } }})
    cargo llvm-cov report --summary-only "${ignore[@]}"
    cargo llvm-cov report --fail-under-lines {{COV_MIN}} "${ignore[@]}"

[group('cover')]
cover-html: _cover-runs
    #!/usr/bin/env bash
    set -euo pipefail
    ignore=({{ if COV_IGNORE == "" { "" } else { "--ignore-filename-regex " + COV_IGNORE } }})
    cargo llvm-cov report --html "${ignore[@]}"
    echo "target/llvm-cov/html/index.html"

[private]
_cover-runs:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo llvm-cov clean --workspace
    cargo llvm-cov --no-report --all-features
    for tier in {{TIERS}}; do
        echo "── SIMD_BACKEND=${tier}"
        SIMD_BACKEND="${tier}" cargo llvm-cov --no-report --all-features
    done
    cargo llvm-cov --no-report --no-default-features

# ── The pull-request gate ─────────────────────────────────────────────────────

# Everything CI will run. Green here means green there.
[group('gate')]
validate: lint deps doc features test-tiers unsafe-check cover
    @echo "{{CRATE}}: validated"

# ── Benchmarks ────────────────────────────────────────────────────────────────

# Run this on the commit you are comparing against, then `just bench` after
# the change. Never compare across sessions.
#
# Record the criterion reference measurement.
[group('bench')]
bench-save NAME="" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just _bench-run "{{NAME}}" --save-baseline before {{ARGS}}

# Compare against the saved baseline.
[group('bench')]
bench NAME="" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just _bench-run "{{NAME}}" --baseline before {{ARGS}}

[private]
_bench-run NAME *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    select=()
    [[ -n "{{NAME}}" ]] && select=(--bench "{{NAME}}")
    "${pin[@]}" cargo bench --all-features "${select[@]}" -- {{ARGS}}

# ── Profiling ─────────────────────────────────────────────────────────────────

# Frame pointers are a rustflag, not a Cargo profile key, so they live here
# rather than in Cargo.toml.
#
# Build with the profiling profile: release codegen a profiler can attribute.
[group('profile')]
profile-perf *ARGS:
    RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling --all-features --all-targets {{ARGS}}

# Criterion's --profile-time runs the benchmark body without its statistical
# machinery, which is what a profiler wants to see.
#
# Profile one criterion bench target under `perf`.
[group('profile')]
perf-bench NAME SECONDS="5" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    bin=$(RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling \
        --all-features --bench "{{NAME}}" --message-format=json-render-diagnostics \
        | jq -r 'select(.executable != null) | .executable' | tail -1)
    [[ -n "${bin}" ]] || { echo "no bench binary for {{NAME}}" >&2; exit 1; }
    pin=()
    [[ -n "${FEC_GOLDEN_CORE:-}" ]] && pin=(taskset -c "${FEC_GOLDEN_CORE}")
    "${pin[@]}" perf record -g --call-graph fp -o perf.data -- \
        "${bin}" --bench --profile-time {{SECONDS}} {{ARGS}}
    echo "perf report -i perf.data"

# ── Release plumbing ──────────────────────────────────────────────────────────

# Cross-crate work lands bottom-up, and the consumer PR pins the exact
# lower-level rev it was built against.
#
# Repoint a git dependency at a landed revision.
[group('release')]
pin DEP REV:
    #!/usr/bin/env bash
    set -euo pipefail
    grep -qE "^{{DEP}} = \{.*rev = \"" Cargo.toml || {
        echo "no single-line git dependency '{{DEP}}' in Cargo.toml; edit it by hand" >&2
        exit 1
    }
    sed -i -E "s|^({{DEP}} = \{.*rev = \")[0-9a-f]{7,40}(\")|\1{{REV}}\2|" Cargo.toml
    grep -E "^{{DEP}} = " Cargo.toml
    cargo update -p {{DEP}}

[group('release')]
publish-dry:
    cargo publish --dry-run

[group('release')]
publish: validate
    cargo publish

# ── Environment ───────────────────────────────────────────────────────────────

# Report whether the toolchain this justfile assumes is present.
[group('meta')]
doctor:
    #!/usr/bin/env bash
    set -uo pipefail
    status=0
    check() {
        if "${@:2}" >/dev/null 2>&1; then
            printf '  ok      %s\n' "$1"
        else
            printf '  MISSING %s\n' "$1"
            status=1
        fi
    }
    echo "{{CRATE}} tooling:"
    check "just"                just --version
    check "cargo"               cargo --version
    check "rustfmt"             cargo fmt --version
    check "clippy"              cargo clippy --version
    check "cargo-llvm-cov"      cargo llvm-cov --version
    check "msrv toolchain {{MSRV}}" rustup run {{MSRV}} cargo --version
    check "jq (perf-bench)"     jq --version
    check "perf (perf-bench)"   perf --version
    check "taskset (bench pinning)" taskset --version
    if [[ -n "{{MIRI}}" ]]; then
        check "nightly miri"    cargo +nightly miri --version
    else
        printf '  n/a     nightly miri (crate forbids unsafe)\n'
    fi
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        printf '  ok      FEC_GOLDEN_CORE=%s\n' "${FEC_GOLDEN_CORE}"
    else
        printf '  unset   FEC_GOLDEN_CORE (benchmarks will not be core-pinned)\n'
    fi
    exit "${status}"
