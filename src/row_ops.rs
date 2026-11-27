//! Internal row helpers: contiguous field row updates, and the value-level
//! row predicates every algebraic decision uses.
//!
//! `fgf` decoders accept every width-correct lane, including noncanonical
//! aliases of prime-field residues, and its zero/unit shortcuts can carry
//! those bytes into results. Raw byte comparisons therefore do not decide
//! field equality; [`is_zero`] and [`rows_equal`] compare decoded values.

use fgf::{Elem, FieldBuffer, FieldElem, ops};

/// Whether every lane of `row` decodes to zero.
pub(crate) fn is_zero<F: FieldBuffer>(row: &[u8]) -> bool {
    row.chunks_exact(F::BYTES)
        .all(|lane| lane.iter().all(|&byte| byte == 0) || F::decode(lane).is_zero())
}

/// Whether two rows of equal length hold the same elements, lane by lane.
pub(crate) fn rows_equal<F: FieldBuffer>(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.chunks_exact(F::BYTES)
            .zip(b.chunks_exact(F::BYTES))
            .all(|(x, y)| x == y || F::decode(x) == F::decode(y))
}

/// Minimum contiguous update assigned to Rayon. The serial and parallel
/// boundary measurements are recorded in `BENCHMARKS.md`.
#[cfg(feature = "parallel")]
pub(crate) const PAR_MIN_BYTES: usize = 2 * 1024 * 1024;

#[cfg(feature = "parallel")]
pub(crate) fn mul_add<F: FieldBuffer>(dst: &mut [u8], factor: Elem<F>, src: &[u8]) {
    use rayon::prelude::{
        IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut,
    };

    if dst.len() < PAR_MIN_BYTES {
        ops::mul_add::<F>(dst, factor, src);
        return;
    }
    let threads = rayon::current_num_threads();
    if threads == 1 {
        ops::mul_add::<F>(dst, factor, src);
        return;
    }
    assert_eq!(dst.len(), src.len(), "mul_add row length mismatch");
    assert!(
        dst.len().is_multiple_of(F::BYTES),
        "mul_add row is not a whole number of lanes"
    );
    let elements = dst.len() / F::BYTES;
    let chunk_elements = elements.div_ceil(threads);
    let chunk_bytes = chunk_elements * F::BYTES;
    dst.par_chunks_mut(chunk_bytes)
        .zip(src.par_chunks(chunk_bytes))
        .for_each(|(dst_chunk, src_chunk)| ops::mul_add::<F>(dst_chunk, factor, src_chunk));
}
#[cfg(not(feature = "parallel"))]
#[inline]
pub(crate) fn mul_add<F: FieldBuffer>(dst: &mut [u8], factor: Elem<F>, src: &[u8]) {
    ops::mul_add::<F>(dst, factor, src);
}
