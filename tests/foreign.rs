//! Differential tests against foreign ground truths, gated on the libraries
//! being present at build time (see `build.rs`). A missing library prints
//! exactly what could not be found and lets the test pass — loudly, never
//! silently.
//!
//! Wired: FLINT's `fq_nmod_mat_lu` over GF(2^8) — FLINT with the modulus
//! 0x11B is byte-compatible with `fgf`'s AES-polynomial `Gf8` encoding —
//! plus the wider fields through the GF(2^8) subfield embedding (rank is
//! preserved under field extension). M4RI's `mzd_echelonize` over GF(2),
//! M4RIE's `mzed_ple` over GF(2^8) under the same `0x11B` field, and
//! FFLAS-FFPACK's `Rank` over the prime field GF(2). The M4RI/M4RIE setters
//! and the FFLAS templates are reached through small shims compiled by
//! `build.rs`.

mod common;

use common::{draw, noise};

#[cfg(not(gfm_flint))]
#[test]
fn flint_absent_loud_skip() {
    eprintln!(
        "FLINT was not found by pkg-config at build time: \
         the fq_nmod_mat_lu differential is SKIPPED. Install flint to run it."
    );
}

#[cfg(not(gfm_m4rie))]
#[test]
fn m4rie_absent_loud_skip() {
    eprintln!(
        "M4RIE was not found by pkg-config at build time: \
         the mzed_ple differential is SKIPPED. Install m4rie to run it."
    );
}

#[cfg(not(gfm_m4ri))]
#[test]
fn m4ri_absent_loud_skip() {
    eprintln!(
        "M4RI was not found by pkg-config at build time: \
         the mzd_echelonize differential is SKIPPED. Install m4ri to run it."
    );
}

#[cfg(not(gfm_fflas))]
#[test]
fn fflas_absent_loud_skip() {
    eprintln!(
        "FFLAS-FFPACK was not found by pkg-config at build time: \
         the FFPACK::Rank differential is SKIPPED. Install fflas-ffpack to run it."
    );
}

#[cfg(gfm_flint)]
mod flint {
    use super::*;
    use fgf::{Field, Gf8, Gf16, Gf32, Gf64};
    use gfm::{Matrix, Ple, PleScratch};
    use std::os::raw::{c_char, c_int, c_long, c_ulong};

    // Layouts probed against the system FLINT (3.6.0) on x86_64. A size or
    // layout mismatch shows up immediately as a wrong answer or a crash, so
    // the differential is self-checking by construction.
    #[repr(C, align(8))]
    struct FqNmodCtx([u64; 20]); // sizeof(fq_nmod_ctx_struct) == 160
    #[repr(C, align(8))]
    struct FqNmodMat([u64; 4]); // sizeof(fq_nmod_mat_struct) == 32
    #[repr(C, align(8))]
    struct NmodPoly([u64; 6]); // sizeof(nmod_poly_struct) == 48

    #[link(name = "flint")]
    unsafe extern "C" {
        fn fq_nmod_ctx_init_modulus(
            ctx: *mut FqNmodCtx,
            modulus: *const NmodPoly,
            var: *const c_char,
        );
        fn fq_nmod_ctx_clear(ctx: *mut FqNmodCtx);
        fn nmod_poly_init(p: *mut NmodPoly, n: c_ulong);
        fn nmod_poly_set_coeff_ui(p: *mut NmodPoly, j: c_long, c: c_ulong);
        fn nmod_poly_clear(p: *mut NmodPoly);
        fn fq_nmod_mat_init(m: *mut FqNmodMat, rows: c_long, cols: c_long, ctx: *const FqNmodCtx);
        fn fq_nmod_mat_clear(m: *mut FqNmodMat, ctx: *const FqNmodCtx);
        fn fq_nmod_mat_entry_set(
            m: *mut FqNmodMat,
            i: c_long,
            j: c_long,
            x: *const NmodPoly,
            ctx: *const FqNmodCtx,
        );
        fn fq_nmod_mat_lu(
            p: *mut c_long,
            m: *mut FqNmodMat,
            rank_check: c_int,
            ctx: *const FqNmodCtx,
        ) -> c_long;
    }

    /// FLINT's rank of a packed GF(2^8) matrix (one byte per element,
    /// polynomial-coefficient encoding, modulus 0x11B).
    fn flint_gf256_rank(data: &[u8], rows: usize, cols: usize) -> i64 {
        unsafe {
            // The AES polynomial x^8 + x^4 + x^3 + x + 1 as an nmod_poly.
            let mut modulus = NmodPoly([0; 6]);
            nmod_poly_init(&mut modulus, 2);
            for bit in [0, 1, 3, 4, 8] {
                nmod_poly_set_coeff_ui(&mut modulus, bit, 1);
            }
            let mut ctx = FqNmodCtx([0; 20]);
            fq_nmod_ctx_init_modulus(&mut ctx, &modulus, c"x".as_ptr());
            let mut mat = FqNmodMat([0; 4]);
            fq_nmod_mat_init(&mut mat, rows as c_long, cols as c_long, &ctx);
            let mut elem = NmodPoly([0; 6]);
            nmod_poly_init(&mut elem, 2);
            for r in 0..rows {
                for c in 0..cols {
                    let v = data[r * cols + c];
                    for bit in 0..8 {
                        nmod_poly_set_coeff_ui(&mut elem, bit, (v >> bit) as c_ulong & 1);
                    }
                    fq_nmod_mat_entry_set(&mut mat, r as c_long, c as c_long, &elem, &ctx);
                }
            }
            let mut perm = vec![0 as c_long; rows.max(1)];
            let rank = fq_nmod_mat_lu(perm.as_mut_ptr(), &mut mat, 0, &ctx);
            fq_nmod_mat_clear(&mut mat, &ctx);
            fq_nmod_ctx_clear(&mut ctx);
            nmod_poly_clear(&mut elem);
            nmod_poly_clear(&mut modulus);
            rank as i64
        }
    }

    /// gfm's rank of a packed matrix over the field `F`.
    fn gfm_rank<F: fgf::FieldKernels>(data: &[u8], rows: usize, cols: usize) -> usize {
        let m = Matrix::<F>::from_rows(rows, cols, data).unwrap();
        Ple::decompose(m, &mut PleScratch::new()).rank()
    }

    #[test]
    fn gf256_rank_matches_flint() {
        let mut state = 0xF117_u64;
        for _case in 0..24 {
            let rows = 1 + draw(&mut state, 24);
            let cols = 1 + draw(&mut state, 24);
            let data = noise(rows * cols, state);
            assert_eq!(
                gfm_rank::<Gf8>(&data, rows, cols) as i64,
                flint_gf256_rank(&data, rows, cols),
                "GF(2^8) rank at {rows}x{cols}",
            );
        }
    }

    /// Embeds a GF(2^8) matrix into the `Gf8` subfield of a tower field: the
    /// subfield elements are exactly those with all tower components but the
    /// first equal to zero.
    fn embed<const W: usize>(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; data.len() * W];
        for (i, &v) in data.iter().enumerate() {
            out[i * W] = v;
        }
        out
    }

    #[test]
    fn subfield_embedding_is_a_homomorphism() {
        // The embedding premise: addition and multiplication agree with the
        // base field on subfield elements.
        let mut state = 0x5EED_u64;
        for _ in 0..50 {
            let x = noise(1, state)[0];
            let y = noise(1, state ^ 1)[0];
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let base = fgf::gf8::Elem(x).mul(fgf::gf8::Elem(y));
            let wide = Gf16::read(&[x, 0]).mul(Gf16::read(&[y, 0]));
            let mut buf = [0u8; 2];
            Gf16::write(&mut buf, wide);
            assert_eq!(buf, [base.0, 0], "GF(2^16) subfield multiplies as GF(2^8)");
            let wide =
                Gf64::read(&[x, 0, 0, 0, 0, 0, 0, 0]).mul(Gf64::read(&[y, 0, 0, 0, 0, 0, 0, 0]));
            let mut buf = [0u8; 8];
            Gf64::write(&mut buf, wide);
            assert_eq!(
                buf,
                [base.0, 0, 0, 0, 0, 0, 0, 0],
                "GF(2^64) subfield multiplies as GF(2^8)"
            );
        }
    }

    #[test]
    fn wide_field_rank_matches_flint_via_subfield() {
        // Rank is preserved under field extension: a GF(2^8) matrix embedded
        // into GF(2^16/32/64) has the same rank, so FLINT's GF(2^8) answer is
        // a ground truth for the wide-field eliminations.
        let mut state = 0xB1DE_u64;
        for _case in 0..12 {
            let rows = 1 + draw(&mut state, 16);
            let cols = 1 + draw(&mut state, 16);
            let data = noise(rows * cols, state);
            let expected = flint_gf256_rank(&data, rows, cols);
            assert_eq!(
                gfm_rank::<Gf16>(&embed::<2>(&data), rows, cols) as i64,
                expected,
                "GF(2^16) rank at {rows}x{cols}",
            );
            assert_eq!(
                gfm_rank::<Gf32>(&embed::<4>(&data), rows, cols) as i64,
                expected,
                "GF(2^32) rank at {rows}x{cols}",
            );
            assert_eq!(
                gfm_rank::<Gf64>(&embed::<8>(&data), rows, cols) as i64,
                expected,
                "GF(2^64) rank at {rows}x{cols}",
            );
        }
    }
}

#[cfg(gfm_m4ri)]
mod m4ri {
    use super::*;
    use gfm::BitMatrix;
    use gfm::bits::{Ple, PleScratch};

    unsafe extern "C" {
        fn gfm_m4ri_rank(packed: *const u64, rows: usize, cols: usize, words: usize) -> usize;
    }

    /// M4RI's rank of a bit matrix, via the compiled shim.
    fn m4ri_rank(packed: &[u64], rows: usize, cols: usize, words: usize) -> usize {
        // SAFETY: `packed` holds `rows * words` words; the shim only reads
        // those and allocates/frees its own M4RI handle.
        unsafe { gfm_m4ri_rank(packed.as_ptr(), rows, cols, words) }
    }

    #[test]
    fn bit_rank_matches_m4ri() {
        let mut st = 0xB17_C0DEu64;
        for _ in 0..24 {
            let rows = 1 + draw(&mut st, 200);
            let cols = 1 + draw(&mut st, 200);
            let bias = 1 + draw(&mut st, 4); // a spread of densities
            let bytes = noise(rows * cols, st);
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let words = cols.div_ceil(64);
            let mut packed = vec![0u64; rows * words];
            for r in 0..rows {
                for c in 0..cols {
                    if (bytes[r * cols + c] as usize).is_multiple_of(bias) {
                        packed[r * words + c / 64] |= 1u64 << (c % 64);
                    }
                }
            }
            let m = BitMatrix::from_rows(rows, cols, &packed).unwrap();
            let gfm = Ple::decompose(m, &mut PleScratch::new()).rank();
            assert_eq!(
                gfm,
                m4ri_rank(&packed, rows, cols, words),
                "rank mismatch vs M4RI at {rows}x{cols}",
            );
        }
    }
}

#[cfg(gfm_m4rie)]
mod m4rie {
    use super::*;
    use fgf::Gf8;
    use gfm::{Matrix, Ple, PleScratch};

    unsafe extern "C" {
        fn gfm_m4rie_gf8_rank(data: *const u8, rows: usize, cols: usize) -> usize;
    }

    /// M4RIE's `mzed_ple` rank of a packed GF(2^8) matrix under `0x11B`.
    fn m4rie_rank(data: &[u8], rows: usize, cols: usize) -> usize {
        // SAFETY: `data` holds `rows * cols` bytes; the shim only reads those
        // and manages its own M4RIE handles.
        unsafe { gfm_m4rie_gf8_rank(data.as_ptr(), rows, cols) }
    }

    #[test]
    fn gf256_rank_matches_m4rie() {
        let mut st = 0x4E1E_9000u64;
        for _ in 0..24 {
            let rows = 1 + draw(&mut st, 130);
            let cols = 1 + draw(&mut st, 130);
            // Full-range GF(2^8) bytes: the 0x11B field makes M4RIE's element
            // encoding identical to fgf's, so this exercises all 256 values.
            let data = noise(rows * cols, st);
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let m = Matrix::<Gf8>::from_rows(rows, cols, &data).unwrap();
            let gfm = Ple::decompose(m, &mut PleScratch::new()).rank();
            assert_eq!(
                gfm,
                m4rie_rank(&data, rows, cols),
                "rank mismatch vs M4RIE at {rows}x{cols}",
            );
        }
    }
}

#[cfg(gfm_fflas)]
mod fflas {
    use super::*;
    use gfm::BitMatrix;
    use gfm::bits::{Ple, PleScratch};

    unsafe extern "C" {
        fn gfm_ffpack_gf2_rank(data: *const u8, m: usize, n: usize) -> usize;
    }

    /// FFLAS-FFPACK's `Rank` over the prime field GF(2).
    fn ffpack_rank(data: &[u8], rows: usize, cols: usize) -> usize {
        // SAFETY: `data` holds `rows * cols` bytes, each 0 or 1; the shim only
        // reads those and owns its own FFLAS allocation.
        unsafe { gfm_ffpack_gf2_rank(data.as_ptr(), rows, cols) }
    }

    #[test]
    fn bit_rank_matches_fflas() {
        let mut st = 0xFF1A_5000u64;
        for _ in 0..24 {
            let rows = 1 + draw(&mut st, 150);
            let cols = 1 + draw(&mut st, 150);
            let bias = 1 + draw(&mut st, 4);
            let src = noise(rows * cols, st);
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // One 0/1 byte per element for FFPACK; the same bits packed for gfm.
            let flat: Vec<u8> = src
                .iter()
                .map(|&b| u8::from((b as usize).is_multiple_of(bias)))
                .collect();
            let words = cols.div_ceil(64);
            let mut packed = vec![0u64; rows * words];
            for r in 0..rows {
                for c in 0..cols {
                    if flat[r * cols + c] == 1 {
                        packed[r * words + c / 64] |= 1u64 << (c % 64);
                    }
                }
            }
            let m = BitMatrix::from_rows(rows, cols, &packed).unwrap();
            let gfm = Ple::decompose(m, &mut PleScratch::new()).rank();
            assert_eq!(
                gfm,
                ffpack_rank(&flat, rows, cols),
                "rank mismatch vs FFLAS-FFPACK at {rows}x{cols}",
            );
        }
    }
}
