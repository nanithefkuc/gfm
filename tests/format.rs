//! The human-readable surfaces render: `Debug` for every public type and
//! `Display` for both error enums, exercised through the public constructors
//! that produce each value.

use fgf::{Gf8B, Gf16};
use gfm::Matrix;
use gfm::bits::{Ple as BitPle, PleScratch as BitPleScratch};
use gfm::dense::{Ple, PleScratch, SmallMatrix};
use gfm::hybrid::Hybrid;
use gfm::incremental::Echelon;

/// Debug output names the type it renders.
fn assert_named<T: core::fmt::Debug>(value: &T, name: &str) {
    let rendered = format!("{value:?}");
    assert!(
        rendered.contains(name),
        "`{name}` missing from its own Debug output: {rendered}"
    );
}

#[test]
fn dense_types_render_debug() {
    let m = Matrix::<Gf8B>::zeros(2, 3).unwrap();
    assert_named(&m, "Matrix");
    assert_named(&m.as_view(), "View");
    let mut owned = m.clone();
    assert_named(&owned.as_view_mut(), "ViewMut");
    let scratch = PleScratch::<Gf8B>::new();
    assert_named(&scratch, "PleScratch");
    let ple = Ple::decompose(m, &mut PleScratch::new());
    assert_named(&ple, "Ple");
    let small = SmallMatrix::<Gf8B, 2>::from_matrix(&Matrix::<Gf8B>::identity(2).unwrap());
    assert_named(&small, "SmallMatrix");
}

#[test]
fn bit_types_render_debug() {
    let bits = gfm::BitMatrix::zeros(2, 3).unwrap();
    assert_named(&bits, "BitMatrix");
    let scratch = BitPleScratch::new();
    assert_named(&scratch, "PleScratch");
    let ple = BitPle::decompose(bits, &mut BitPleScratch::new());
    assert_named(&ple, "Ple");
}

#[test]
fn hybrid_types_render_debug() {
    let one = <Gf8B as fgf::Field>::Elem::ONE;
    let mut hybrid = Hybrid::<Gf8B>::new(2, 1);
    hybrid.push_field_row(&[0], &[one], &[3]);
    hybrid.push_field_row(&[1], &[one], &[4]);
    let solution = hybrid.solve().unwrap();
    assert_named(&solution, "Solution");
}

#[test]
fn incremental_types_render_debug() {
    let mut echelon = Echelon::<Gf8B>::new(2, 1, true).unwrap();
    assert_named(&echelon, "Echelon");
    echelon.absorb_unit(0, &[7]);
    let retained = echelon.retained_rows().next().unwrap();
    assert_named(&retained, "RetainedRow");
}

#[test]
fn geometry_errors_display() {
    let overflow = Matrix::<Gf16>::zeros(usize::MAX, usize::MAX).unwrap_err();
    assert!(overflow.to_string().contains("overflows"));
    let ragged = Matrix::<Gf16>::from_rows(2, 2, &[0, 1, 2]).unwrap_err();
    assert!(ragged.to_string().contains("whole number"));
    let shape = Matrix::<Gf8B>::from_rows(2, 2, &[0, 1, 2]).unwrap_err();
    assert!(shape.to_string().contains("do not compose"));
    // The error types ride the standard error trait where std is available.
    #[cfg(feature = "std")]
    {
        let surfaced: &dyn std::error::Error = &shape;
        assert!(surfaced.source().is_none());
    }
}

#[test]
fn solve_errors_display() {
    let inconsistent = {
        let mut hybrid = Hybrid::<Gf8B>::new(2, 1);
        hybrid.push_binary_row(&[], &[1]);
        hybrid.solve().unwrap_err()
    };
    assert!(inconsistent.to_string().contains("inconsistent"));
    let singular = {
        let m = Matrix::<Gf8B>::from_rows(2, 2, &[1, 0, 0, 0]).unwrap();
        Ple::decompose(m, &mut PleScratch::new())
            .inverse_into(&mut Matrix::<Gf8B>::zeros(2, 2).unwrap())
            .unwrap_err()
    };
    assert!(singular.to_string().contains("singular"));
    #[cfg(feature = "std")]
    {
        let surfaced: &dyn std::error::Error = &inconsistent;
        assert!(surfaced.source().is_none());
    }
}
