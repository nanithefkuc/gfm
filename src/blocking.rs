//! Dense-elimination blocking policy.
//!
//! Blocking is applied only where a measurement selected it; every other
//! backend and shape uses the unblocked sweep. The comparison and the retained
//! thresholds are recorded in `BENCHMARKS.md`. Backend selection itself is
//! `fgf`'s: this module reads the resolved backend and adds no detector,
//! override, or cache.

use fgf::{Backend, FieldBuffer};

/// Dense-elimination panel width for the active shape and backend.
pub(crate) fn panel_width<F: FieldBuffer>(rows: usize, cols: usize) -> usize {
    if fgf::backend_for::<F>() != Backend::V3GfniCrypto {
        return 1;
    }
    let order = rows.min(cols);
    match F::BYTES {
        1 if order <= 32 => 64,
        2 if order <= 64 => 64,
        _ => 1,
    }
}
