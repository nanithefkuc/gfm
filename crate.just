# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to gfm belong in this file and are documented in `AGENTS.md`.

CRATE := 'gfm'
MSRV := '1.93'

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
