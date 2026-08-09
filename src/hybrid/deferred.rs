//! The deferred row-operation log.
//!
//! The sparse phase decides the whole pivot structure from coefficients alone,
//! which are cheap to combine. The expensive work is the payload: every
//! elimination implies `dst.rhs ^= factor · src.rhs` over `sym_len` bytes.
//! RFC 6330 §5.4.2.1 is explicit that this is where the time goes and that the
//! operations "should not be performed until the affected row is itself
//! chosen". So each coefficient elimination is *recorded* here; the payloads
//! are touched once, at the end, and only for the rows the answer actually
//! reads — redundant received rows are never paid for.

use alloc::vec::Vec;

use fgf::FieldKernels;

/// One recorded elimination: `row[dst] += factor · row[src]`.
#[derive(Clone, Copy)]
pub(crate) struct Op<F: FieldKernels> {
    pub dst: u32,
    pub src: u32,
    pub factor: F::Elem,
}

/// An ordered log of eliminations, replayed against payloads on demand.
pub(crate) struct DeferredLog<F: FieldKernels> {
    ops: Vec<Op<F>>,
}

impl<F: FieldKernels> DeferredLog<F> {
    pub(crate) fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub(crate) fn clear(&mut self) {
        self.ops.clear();
    }

    pub(crate) fn record(&mut self, dst: u32, src: u32, factor: F::Elem) {
        self.ops.push(Op { dst, src, factor });
    }

    pub(crate) fn ops(&self) -> &[Op<F>] {
        &self.ops
    }
}

impl<F: FieldKernels> core::fmt::Debug for DeferredLog<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeferredLog")
            .field("ops", &self.ops.len())
            .finish()
    }
}
