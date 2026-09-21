//! Public-surface test for the backend seam.
//!
//! `gfm::backend_for` is a pass-through to `fgf`'s per-field resolution; the
//! property worth pinning is that gfm reports exactly the backend the field
//! kernels will run on, and that the stack-wide `SIMD_BACKEND` override
//! reaches it. There is no second resolver to drift.

use fgf::{Gf8B, Gf16, Gf32, Gf64};
mod common;
use gfm::internals::PleInternals;
use gfm::{Backend, Matrix, backend_for};

#[test]
fn reports_the_backend_the_kernels_run_on() {
    assert_eq!(backend_for::<Gf8B>(), fgf::backend_for::<Gf8B>());
    assert_eq!(backend_for::<Gf16>(), fgf::backend_for::<Gf16>());
    assert_eq!(backend_for::<Gf32>(), fgf::backend_for::<Gf32>());
    assert_eq!(backend_for::<Gf64>(), fgf::backend_for::<Gf64>());
}

#[test]
fn scalar_override_forces_scalar() {
    // The override is read once per process, so this arm only fires when the
    // suite itself is run under `SIMD_BACKEND=scalar` (as CI's backend sweep
    // does). Without the variable there is nothing to assert beyond the
    // pass-through above.
    if std::env::var("SIMD_BACKEND")
        .ok()
        .is_some_and(|v| v == "scalar")
    {
        assert_eq!(backend_for::<Gf8B>(), Backend::Scalar);
        assert_eq!(backend_for::<Gf64>(), Backend::Scalar);
    }
}

#[test]
fn backend_fingerprint_is_stable() {
    use fgf::Field;
    use gfm::{Matrix, Ple, PleScratch};
    use std::io::Write;

    writeln!(
        std::io::stderr().lock(),
        "backend_selection requested={:?} gf8b={:?} gf16={:?} gf32={:?} gf64={:?}",
        std::env::var("SIMD_BACKEND").ok(),
        backend_for::<Gf8B>(),
        backend_for::<Gf16>(),
        backend_for::<Gf32>(),
        backend_for::<Gf64>(),
    )
    .expect("write backend report");

    fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
        for &byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x100_0000_01B3);
        }
    }

    let mut state = 0xBACC_E11D_5EED_u64;
    let mut matrix = Matrix::<Gf8B>::zeros(160, 160).unwrap();
    for row in 0..160 {
        for col in 0..160 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            matrix.set(row, col, Gf8B::decode(&[(state >> 56) as u8]));
        }
    }
    let ple = Ple::decompose(matrix, &mut PleScratch::new());
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    hash_bytes(&mut hash, &(ple.rank() as u64).to_le_bytes());
    for row in 0..ple.rows() {
        hash_bytes(&mut hash, ple.lu().row(row));
    }
    for perm in [ple.p(), ple.q()] {
        let mut image: Vec<usize> = (0..perm.len()).collect();
        perm.apply(&mut image);
        for index in image {
            hash_bytes(&mut hash, &(index as u64).to_le_bytes());
        }
    }
    for profile in [ple.row_rank_profile(), ple.col_rank_profile()] {
        for &index in profile {
            hash_bytes(&mut hash, &(index as u64).to_le_bytes());
        }
    }
    assert_eq!(hash, 0xD3F8_C323_33AA_D204);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_row_dispatch_matches_field_kernel() {
    use fgf::Field;
    use gfm::internals::row_ops::mul_add as row_mul_add;

    for bytes in [2 * 1024 * 1024 - 1, 2 * 1024 * 1024, 4 * 1024 * 1024] {
        let src = vec![0xA7; bytes];
        let mut expected = vec![0x39; bytes];
        let mut actual = expected.clone();
        let factor = Gf8B::decode(&[0x53]);
        fgf::ops::mul_add::<Gf8B>(&mut expected, factor, &src);
        row_mul_add::<Gf8B>(&mut actual, factor, &src);
        assert_eq!(actual, expected, "row bytes {bytes}");
    }
}

/// A deterministic noise matrix with shape deliberately awkward for panel
/// arithmetic.
fn noise_matrix(rows: usize, cols: usize, seed: u64) -> Matrix<Gf8B> {
    use common::noise;
    Matrix::from_rows(rows, cols, &noise(rows * cols, seed)).unwrap()
}

#[test]
fn panel_width_does_not_change_the_decomposition() {
    use gfm::{Ple, PleScratch};

    let a = noise_matrix(37, 41, 0x11);
    let reference = Ple::decompose(a.clone(), &mut PleScratch::new());
    for width in [1, 2, 7, 16, 64] {
        let ple = Ple::decompose_with_panel_width(a.clone(), &mut PleScratch::new(), width);
        assert_eq!(ple.rank(), reference.rank(), "rank at width {width}");
        assert_eq!(ple.lu(), reference.lu(), "LU at width {width}");
        assert_eq!(ple.p(), reference.p(), "row perm at width {width}");
        assert_eq!(ple.q(), reference.q(), "col perm at width {width}");
    }
}

#[test]
fn newton_john_trailing_update_matches_the_dispatched_one() {
    use gfm::{Ple, PleScratch};

    let a = noise_matrix(33, 33, 0x22);
    let reference = Ple::decompose(a.clone(), &mut PleScratch::new());
    let nj = Ple::decompose_newton_john(a, &mut PleScratch::new());
    assert_eq!(nj.rank(), reference.rank());
    assert_eq!(nj.lu(), reference.lu());
    assert_eq!(nj.p(), reference.p());
    assert_eq!(nj.q(), reference.q());
}

#[test]
fn bit_ple_plain_and_m4ri_agree() {
    use gfm::BitMatrix;
    use gfm::bits::{Ple as BitPle, PleScratch as BitPleScratch};
    use gfm::internals::{BitMatrixInternals, BitPleInternals};

    // 61 x 67 straddles word and table boundaries.
    let (rows, cols) = (61usize, 67usize);
    let bits = common::noise(rows * cols, 0x44);
    let mut a = BitMatrix::zeros(rows, cols).unwrap();
    for (i, bit) in bits.iter().enumerate() {
        if bit & 1 == 1 {
            a.set(i / cols, i % cols, true);
        }
    }
    // Touch the physical-layout inspection so the facade stays exercised.
    assert_eq!(a.physical_row_index(0), 0);
    let reference = BitPle::decompose(a.clone(), &mut BitPleScratch::new());
    let plain = BitPle::decompose_plain(a.clone(), &mut BitPleScratch::new());
    let m4ri = BitPle::decompose_m4ri(a, &mut BitPleScratch::new());
    for (label, ple) in [("plain", &plain), ("m4ri", &m4ri)] {
        assert_eq!(ple.rank(), reference.rank(), "{label} rank");
        assert_eq!(ple.lu(), reference.lu(), "{label} LU");
        assert_eq!(ple.p(), reference.p(), "{label} row perm");
        assert_eq!(ple.q(), reference.q(), "{label} col perm");
    }
}
