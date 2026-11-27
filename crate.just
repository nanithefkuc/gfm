# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to gfm belong in this file and are documented in `AGENTS.md`.

CRATE := 'gfm'
MSRV := '1.93'
API_NIGHTLY := 'nightly-2026-09-30'

# Stable public-surface gate. `API_PATHS_MODE` is `enforce`: every stable item
# has exactly one path. The inventory is generated from the portable, default,
# and all-stable-feature configurations on x86-64 and committed under `api/`.
API_PUBLIC_API := '0.52.0'
API_FORMAT := '61'
API_TARGETS := 'x86_64-unknown-linux-gnu'
API_PATHS_MODE := 'enforce'

# Every Cargo feature except `default` and `internals`; `_api-generate` fails
# when it disagrees with the manifest, so a new feature cannot drop out of the
# `all` inventory cell.
API_STABLE_FEATURES := 'parallel,simd,simd512,std'

# Backend tiers fgf 4.0.1 routes on x86-64, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process; the runs use all features, so `simd512` makes `v4x` reachable.
# These are requests: unsupported tiers fall back, and a run proves only the
# backend it reports.
TIERS := 'v4x v3_gfni_crypto v3 v2 scalar'

# `cargo miri test` argument sets, one per line. Empty when the crate declares
# no unsafe surface, in which case `just unsafe-check` reports and skips.
MIRI := ''

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
COV_IGNORE := ''


# Generate the stable inventories, their metadata, and the one-path report
# into `target/api/generated`. Every configuration builds unchanged and no
# inventory line is filtered: the `all` cell builds every stable feature, and
# a separate `--all-features` rustdoc document must have the same stable export
# path set. The rustdoc JSON is deleted before each run and copied to
# `$out/json` after it, so a skipped rustdoc pass fails instead of reusing a
# stale document. Each inventory closes with the external re-export section,
# and the metadata records the resolved version of every re-exported
# dependency. Tool versions are checked before anything is generated.
[private]
_api-generate:
    #!/usr/bin/env bash
    set -euo pipefail
    have=$(cargo public-api --version 2>/dev/null || true)
    [[ "$have" == "cargo-public-api {{API_PUBLIC_API}}" ]] || {
        printf 'api: need cargo-public-api %s, found "%s"; run: cargo binstall cargo-public-api@%s\n' "{{API_PUBLIC_API}}" "$have" "{{API_PUBLIC_API}}" >&2
        exit 2
    }
    API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" python3 -m unittest -q api/test_surface.py
    declared=$(cargo metadata --no-deps --format-version 1 \
        | jq -r '.packages[0].features | keys - ["default", "internals"] | sort | join(",")')
    pinned=$(tr ',' '\n' <<<"{{API_STABLE_FEATURES}}" | sort | paste -sd,)
    [[ "$declared" == "$pinned" ]] || {
        printf 'api: API_STABLE_FEATURES is %s but the manifest declares %s\n' "$pinned" "$declared" >&2
        exit 2
    }
    out=target/api/generated
    rm -rf "$out"
    mkdir -p "$out/inventory" "$out/json"
    export RUSTDOCFLAGS="-Z unstable-options --document-hidden-items"
    configs=("portable|--no-default-features" "default|" "all|--no-default-features --features {{API_STABLE_FEATURES}}")
    {
        printf 'nightly %s\n' "{{API_NIGHTLY}}"
        printf 'rustdoc-json-format %s\n' "{{API_FORMAT}}"
        printf 'cargo-public-api %s (-s: blanket impls omitted)\n' "{{API_PUBLIC_API}}"
        printf 'rustdocflags %s\n' "$RUSTDOCFLAGS"
        printf 'targets %s\n' "{{API_TARGETS}}"
        for entry in "${configs[@]}"; do flags="${entry#*|}"; printf 'config %s%s\n' "${entry%%|*}" "${flags:+ $flags}"; done
    } >"$out/metadata.txt"
    failed=0
    for target in {{API_TARGETS}}; do
        for entry in "${configs[@]}"; do
            config="${entry%%|*}"
            # shellcheck disable=SC2206
            flags=(${entry#*|})
            doc="target/api/$target/doc/gfm.json"
            rm -f "$doc"
            cargo +{{API_NIGHTLY}} public-api -s --color never --target "$target" \
                --target-dir target/api "${flags[@]}" >"$out/inventory/$config-$target.txt" \
                2>"$out/$config-$target.log" || {
                cat "$out/$config-$target.log" >&2
                exit 1
            }
            [[ -f "$doc" ]] || {
                printf 'api: %s %s: rustdoc JSON missing after the run\n' "$config" "$target" >&2
                exit 1
            }
            cp "$doc" "$out/json/$config-$target.json"
            printf '# external re-exports\n' >>"$out/inventory/$config-$target.txt"
            python3 api/surface.py reexports --format {{API_FORMAT}} \
                "$out/json/$config-$target.json" >>"$out/inventory/$config-$target.txt"
            if ! report=$(python3 api/surface.py paths --format {{API_FORMAT}} \
                "$out/json/$config-$target.json" --mode {{API_PATHS_MODE}}); then
                printf '%s\n' "$report"
                failed=1
            fi
            printf 'api: %-8s %-26s %s\n' "$config" "$target" "$(head -n 1 <<<"$report" | sed 's/^surface: //')"
            [[ "$config-$target" == all-x86_64-unknown-linux-gnu ]] && printf '%s\n' "$report" >"$out/paths.txt"
        done
        doc="target/api/$target/doc/gfm.json"
        rm -f "$doc"
        cargo +{{API_NIGHTLY}} rustdoc --lib --all-features --target "$target" \
            --target-dir target/api -- -Z unstable-options --output-format json --document-hidden-items \
            2>"$out/all-features-$target.log" || {
            cat "$out/all-features-$target.log" >&2
            exit 1
        }
        [[ -f "$doc" ]] || {
            printf 'api: all-features %s: rustdoc JSON missing after the run\n' "$target" >&2
            exit 1
        }
        cp "$doc" "$out/json/all-features-$target.json"
        python3 api/surface.py stable-paths-equal --format {{API_FORMAT}} \
            "$out/json/all-features-$target.json" "$out/json/all-$target.json" || failed=1
    done
    metadata=$(cargo metadata --format-version 1)
    while IFS= read -r dep; do
        # Versions come from gfm's own normal dependency edges, matched on the
        # rustdoc crate name, so another copy elsewhere in the graph is ignored.
        versions=$(jq -r --arg name "$dep" '
            .resolve.root as $root
            | (.resolve.nodes[] | select(.id == $root) | .deps[]
               | select(any(.dep_kinds[]; .kind == null)) | select(.name == $name) | .pkg) as $pkg
            | .packages[] | select(.id == $pkg) | .version' <<<"$metadata" | sort -u)
        [[ -n "$versions" && $(wc -l <<<"$versions") -eq 1 ]] || {
            printf 'api: re-exported crate %s does not resolve to exactly one normal dependency of gfm (found: %s)\n' "$dep" "${versions:-none}" >&2
            exit 1
        }
        printf 'dependency %s %s\n' "$dep" "$versions"
    done < <(grep -h '^reexport ' "$out"/inventory/*.txt | sed 's/^reexport [^=]*= //' | cut -d: -f1 | sort -u) >>"$out/metadata.txt"
    if [[ "{{API_PATHS_MODE}}" == report ]]; then cat "$out/paths.txt"; else tail -n 1 "$out/paths.txt"; fi
    for entry in "${configs[@]}"; do
        config="${entry%%|*}"
        files=()
        for target in {{API_TARGETS}}; do files+=("$out/inventory/$config-$target.txt"); done
        python3 api/surface.py same "$config" "${files[@]}" || failed=1
    done
    exit "$failed"

# Check the committed stable API inventory: regenerate every inventory and
# fail on any difference from `api/`. Read-only.
[group('gate')]
api: _api-generate
    #!/usr/bin/env bash
    set -euo pipefail
    out=target/api/generated
    if ! diff -u api/metadata.txt "$out/metadata.txt" || ! diff -ru api/inventory "$out/inventory"; then
        printf 'api: the public surface differs from the committed inventory; an authorized API change runs `just api-update`\n' >&2
        exit 1
    fi
    echo "api: stable surface matches the committed inventory"

# Rewrite the committed inventory. Run only inside an explicitly authorized
# API change; review the resulting diff.
[group('gate')]
api-update: _api-generate
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf api/inventory
    cp -r target/api/generated/inventory api/inventory
    cp target/api/generated/metadata.txt api/metadata.txt
    echo "api: inventory rewritten; review git diff -- api/"
