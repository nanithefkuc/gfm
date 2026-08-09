//! Public-surface test for the backend seam.
//!
//! `gfm::backend_for` is a pass-through to `fgf`'s per-field resolution; the
//! property worth pinning is that gfm reports exactly the backend the field
//! kernels will run on, and that the stack-wide `SIMD_BACKEND` override
//! reaches it. There is no second resolver to drift.

use fgf::{Gf8, Gf16, Gf32, Gf64};
use gfm::{Backend, backend_for};

#[test]
fn reports_the_backend_the_kernels_run_on() {
    assert_eq!(backend_for::<Gf8>(), fgf::backend_for::<Gf8>());
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
        assert_eq!(backend_for::<Gf8>(), Backend::Scalar);
        assert_eq!(backend_for::<Gf64>(), Backend::Scalar);
    }
}

#[cfg(feature = "internals")]
#[test]
fn backend_fingerprint_is_stable() {
    use fgf::Field;
    use gfm::{Matrix, Ple, PleScratch};

    fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
        for &byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x100_0000_01B3);
        }
    }

    let mut state = 0xBACC_E11D_5EED_u64;
    let mut matrix = Matrix::<Gf8>::zeros(160, 160).unwrap();
    for row in 0..160 {
        for col in 0..160 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            matrix.set(row, col, Gf8::read(&[(state >> 56) as u8]));
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

#[cfg(all(feature = "parallel", feature = "internals"))]
#[test]
fn parallel_row_dispatch_matches_field_kernel() {
    use fgf::Field;

    for bytes in [2 * 1024 * 1024 - 1, 2 * 1024 * 1024, 4 * 1024 * 1024] {
        let src = vec![0xA7; bytes];
        let mut expected = vec![0x39; bytes];
        let mut actual = expected.clone();
        let factor = Gf8::read(&[0x53]);
        fgf::ops::mul_add::<Gf8>(&mut expected, factor, &src);
        gfm::benchmark_mul_add::<Gf8>(&mut actual, factor, &src);
        assert_eq!(actual, expected, "row bytes {bytes}");
    }
}
