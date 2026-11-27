//! The deferred row-operation log.
//!
//! The sparse phase decides the whole pivot structure from coefficients alone,
//! which are cheap to combine. The expensive work is the payload: every
//! elimination implies `dst.rhs += factor · src.rhs` over `sym_len` bytes.
//! RFC 6330 §5.4.2.1 is explicit that this is where the time goes and that the
//! operations "should not be performed until the affected row is itself
//! chosen". So each coefficient elimination is *recorded* here; the payloads
//! are touched once, at the end, and only for the rows the answer actually
//! reads — redundant received rows are never paid for.

use alloc::vec::Vec;

use fgf::{Elem, FieldBuffer};

/// One recorded elimination: `row[dst] += factor · row[src]`.
#[derive(Clone, Copy)]
pub(crate) struct Op<F: FieldBuffer> {
    pub dst: u32,
    pub src: u32,
    pub factor: Elem<F>,
}

/// An ordered log of eliminations, replayed against payloads on demand.
pub(crate) struct DeferredLog<F: FieldBuffer> {
    ops: Vec<Op<F>>,
}

impl<F: FieldBuffer> DeferredLog<F> {
    pub(crate) fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub(crate) fn clear(&mut self) {
        self.ops.clear();
    }

    pub(crate) fn record(&mut self, dst: u32, src: u32, factor: Elem<F>) {
        self.ops.push(Op { dst, src, factor });
    }

    pub(crate) fn ops(&self) -> &[Op<F>] {
        &self.ops
    }
}

impl<F: FieldBuffer> core::fmt::Debug for DeferredLog<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeferredLog")
            .field("ops", &self.ops.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use fgf::{
        FiniteField,
        binary::{AES, Polynomial},
    };

    use super::*;

    type Gf8B = FiniteField<2, 8, Polynomial<AES>>;

    #[test]
    fn debug_names_the_log() {
        let mut log = DeferredLog::<Gf8B>::new();
        log.record(1, 0, Elem::<Gf8B>::ONE);
        assert!(format!("{log:?}").contains("DeferredLog"));
    }
}
