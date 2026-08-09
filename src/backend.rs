//! Backend selection for dispatch decisions.
//!
//! `gfm` owns no SIMD kernels: the crate is `#![forbid(unsafe_code)]` and
//! every vector kernel is reached through `fgf::ops`. This module exists for
//! exactly one reason — the per-(field, backend) blocking factor and panel
//! width the elimination uses key off the backend those kernels will
//! actually run on.
//!
//! Detection, ordering, and the downgrade-only `SIMD_BACKEND` override are
//! single-source in `simdispatch`; [`Backend`] is its ladder, re-exported
//! here through `fgf`, so `gfm::Backend`, `fgf::Backend`, and every other
//! consumer's `Backend` are the same type. Nothing in this module probes the
//! host, reads the environment, or caches a value, and no `gfm`-specific
//! override variable exists.

pub use fgf::Backend;
use fgf::FieldKernels;

/// The backend `fgf`'s kernels for field `F` resolve to on this host.
///
/// This is the value gfm's per-(field, backend) dispatch decisions key on:
/// the backend the row kernels will actually execute, already narrowed per
/// field (wider fields report [`Backend::Scalar`] where they have no vector
/// kernels) and already adjusted by the downgrade-only `SIMD_BACKEND`
/// override. `gfm` does not narrow it further; there is no second resolver.
#[inline]
#[must_use]
pub fn backend_for<F: FieldKernels>() -> Backend {
    fgf::backend_for::<F>()
}
/// Measured dense-elimination panel width for the active shape and backend.
///
/// The A/B twin is panel width one. Only this host's `V3GfniCrypto` kernels
/// have measured blocking decisions; every unmeasured backend keeps the plain
/// A/B measurements and retained thresholds are recorded in `BENCHMARKS.md`.
pub(crate) fn panel_width<F: FieldKernels>(rows: usize, cols: usize) -> usize {
    if backend_for::<F>() != Backend::V3GfniCrypto {
        return 1;
    }
    let order = rows.min(cols);
    match F::BYTES {
        1 if order <= 32 => 64,
        2 if order <= 64 => 64,
        _ => 1,
    }
}
