//! Internal dispatch for contiguous field row updates.

use fgf::{FieldKernels, ops};

/// Minimum contiguous update assigned to Rayon. The serial and parallel
/// boundary measurements are recorded in `BENCHMARKS.md`.
#[cfg(feature = "parallel")]
pub(crate) const PAR_MIN_BYTES: usize = 2 * 1024 * 1024;

#[cfg(feature = "parallel")]
pub(crate) fn mul_add<F: FieldKernels>(dst: &mut [u8], factor: F::Elem, src: &[u8]) {
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
    let elements = dst.len() / F::BYTES;
    let mut factor_bytes = [0u8; 8];
    F::encode(&mut factor_bytes[..F::BYTES], factor);
    let chunk_elements = elements.div_ceil(threads);
    let chunk_bytes = chunk_elements * F::BYTES;
    dst.par_chunks_mut(chunk_bytes)
        .zip(src.par_chunks(chunk_bytes))
        .for_each(|(dst_chunk, src_chunk)| {
            let factor = F::decode(&factor_bytes[..F::BYTES]);
            ops::mul_add::<F>(dst_chunk, factor, src_chunk);
        });
}
#[cfg(not(feature = "parallel"))]
#[inline]
pub(crate) fn mul_add<F: FieldKernels>(dst: &mut [u8], factor: F::Elem, src: &[u8]) {
    ops::mul_add::<F>(dst, factor, src);
}
