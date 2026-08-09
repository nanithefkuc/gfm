//! Same-host rank comparison against the installed FLINT, M4RI, M4RIE, and
//! FFLAS-FFPACK libraries. Each timed path constructs its owned matrix from the
//! same immutable input and computes rank, so allocation and import costs are
//! included for every implementation.

use core::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fgf::Gf8;
use gfm::bits::{Ple as BitPle, PleScratch as BitPleScratch};
use gfm::{BitMatrix, Matrix, Ple, PleScratch};
#[cfg(gfm_flint)]
use std::os::raw::{c_char, c_int, c_long, c_ulong};

#[cfg(gfm_m4ri)]
unsafe extern "C" {
    fn gfm_m4ri_rank(packed: *const u64, rows: usize, cols: usize, words: usize) -> usize;
}

#[cfg(gfm_m4rie)]
unsafe extern "C" {
    fn gfm_m4rie_gf8_rank(data: *const u8, rows: usize, cols: usize) -> usize;
}

#[cfg(gfm_fflas)]
unsafe extern "C" {
    fn gfm_ffpack_gf2_rank(data: *const u8, rows: usize, cols: usize) -> usize;
}

#[cfg(gfm_flint)]
#[repr(C, align(8))]
struct FqNmodCtx([u64; 20]);
#[cfg(gfm_flint)]
#[repr(C, align(8))]
struct FqNmodMat([u64; 4]);
#[cfg(gfm_flint)]
#[repr(C, align(8))]
struct NmodPoly([u64; 6]);

#[cfg(gfm_flint)]
#[link(name = "flint")]
unsafe extern "C" {
    fn fq_nmod_ctx_init_modulus(ctx: *mut FqNmodCtx, modulus: *const NmodPoly, var: *const c_char);
    fn fq_nmod_ctx_clear(ctx: *mut FqNmodCtx);
    fn nmod_poly_init(p: *mut NmodPoly, n: c_ulong);
    fn nmod_poly_set_coeff_ui(p: *mut NmodPoly, j: c_long, c: c_ulong);
    fn nmod_poly_clear(p: *mut NmodPoly);
    fn fq_nmod_mat_init(m: *mut FqNmodMat, rows: c_long, cols: c_long, ctx: *const FqNmodCtx);
    fn fq_nmod_mat_clear(m: *mut FqNmodMat, ctx: *const FqNmodCtx);
    fn fq_nmod_mat_entry_set(
        m: *mut FqNmodMat,
        row: c_long,
        col: c_long,
        value: *const NmodPoly,
        ctx: *const FqNmodCtx,
    );
    fn fq_nmod_mat_lu(
        permutation: *mut c_long,
        matrix: *mut FqNmodMat,
        rank_check: c_int,
        ctx: *const FqNmodCtx,
    ) -> c_long;
}

#[cfg(gfm_flint)]
fn flint_gf8_rank(data: &[u8], rows: usize, cols: usize) -> usize {
    // SAFETY: the opaque layouts match the installed FLINT 3.6.0 ABI, every
    // object is initialized before use and cleared before return, and `data`
    // contains exactly `rows * cols` elements.
    unsafe {
        let mut modulus = NmodPoly([0; 6]);
        nmod_poly_init(&mut modulus, 2);
        for bit in [0, 1, 3, 4, 8] {
            nmod_poly_set_coeff_ui(&mut modulus, bit, 1);
        }
        let mut context = FqNmodCtx([0; 20]);
        fq_nmod_ctx_init_modulus(&mut context, &modulus, c"x".as_ptr());
        let mut matrix = FqNmodMat([0; 4]);
        fq_nmod_mat_init(&mut matrix, rows as c_long, cols as c_long, &context);
        let mut element = NmodPoly([0; 6]);
        nmod_poly_init(&mut element, 2);
        for row in 0..rows {
            for col in 0..cols {
                let value = data[row * cols + col];
                for bit in 0..8 {
                    nmod_poly_set_coeff_ui(&mut element, bit, c_ulong::from(value >> bit) & 1);
                }
                fq_nmod_mat_entry_set(
                    &mut matrix,
                    row as c_long,
                    col as c_long,
                    &element,
                    &context,
                );
            }
        }
        let mut permutation = vec![0 as c_long; rows.max(1)];
        let rank = fq_nmod_mat_lu(permutation.as_mut_ptr(), &mut matrix, 0, &context);
        fq_nmod_mat_clear(&mut matrix, &context);
        fq_nmod_ctx_clear(&mut context);
        nmod_poly_clear(&mut element);
        nmod_poly_clear(&mut modulus);
        rank as usize
    }
}

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

fn bit_inputs(n: usize) -> (Vec<u64>, Vec<u8>) {
    let words = n.div_ceil(64);
    let mut packed = vec![0u64; n * words];
    let mut flat = vec![0u8; n * n];
    let mut state = 0xB17C_0DE0 ^ n as u64;
    for row in 0..n {
        for col in 0..n {
            let bit = (next(&mut state) >> 63) as u8;
            flat[row * n + col] = bit;
            packed[row * words + col / 64] |= u64::from(bit) << (col % 64);
        }
    }
    (packed, flat)
}

fn gf8_input(n: usize) -> Vec<u8> {
    let mut state = 0x4E1E_9000 ^ n as u64;
    (0..n * n).map(|_| (next(&mut state) >> 56) as u8).collect()
}

fn bench_gf2(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitors_gf2_rank");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for n in [128usize, 256, 512] {
        let (packed, flat) = bit_inputs(n);
        let words = n.div_ceil(64);
        #[cfg(not(gfm_m4ri))]
        let _ = words;
        #[cfg(not(gfm_fflas))]
        let _ = &flat;
        group.bench_with_input(BenchmarkId::new("gfm", n), &n, |b, _| {
            let mut scratch = BitPleScratch::new();
            b.iter(|| {
                let matrix = BitMatrix::from_rows(n, n, black_box(&packed)).unwrap();
                black_box(BitPle::decompose(matrix, &mut scratch).rank());
            });
        });
        #[cfg(gfm_m4ri)]
        group.bench_with_input(BenchmarkId::new("m4ri", n), &n, |b, _| {
            b.iter(|| {
                // SAFETY: `packed` contains exactly `n * words` words and the
                // shim reads them for the duration of this call.
                black_box(unsafe { gfm_m4ri_rank(packed.as_ptr(), n, n, words) });
            });
        });
        #[cfg(gfm_fflas)]
        group.bench_with_input(BenchmarkId::new("fflas_ffpack", n), &n, |b, _| {
            b.iter(|| {
                // SAFETY: `flat` contains exactly `n * n` binary bytes and the
                // shim reads them for the duration of this call.
                black_box(unsafe { gfm_ffpack_gf2_rank(flat.as_ptr(), n, n) });
            });
        });
    }
    group.finish();
}

fn bench_gf8(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitors_gf8_rank");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for n in [128usize, 256, 512] {
        let data = gf8_input(n);
        group.bench_with_input(BenchmarkId::new("gfm", n), &n, |b, _| {
            let mut scratch = PleScratch::new();
            b.iter(|| {
                let matrix = Matrix::<Gf8>::from_rows(n, n, black_box(&data)).unwrap();
                black_box(Ple::decompose(matrix, &mut scratch).rank());
            });
        });
        #[cfg(gfm_m4rie)]
        group.bench_with_input(BenchmarkId::new("m4rie", n), &n, |b, _| {
            b.iter(|| {
                // SAFETY: `data` contains exactly `n * n` bytes and the shim
                // reads them for the duration of this call.
                black_box(unsafe { gfm_m4rie_gf8_rank(data.as_ptr(), n, n) });
            });
        });
        #[cfg(gfm_flint)]
        group.bench_with_input(BenchmarkId::new("flint", n), &n, |b, _| {
            b.iter(|| black_box(flint_gf8_rank(black_box(&data), n, n)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gf2, bench_gf8);
criterion_main!(benches);
