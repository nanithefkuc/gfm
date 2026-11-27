//! Reporting surfaces: error `Display` output carries the offending values
//! each variant stores, and `Debug` summaries carry the shape facts a caller
//! needs to identify a value. Wording is not part of the contract.

use fgf::FiniteField;
use fgf::binary::{AES, Polynomial};
use gfm::bits::{BitMatrix, Ple as BitPle, PleScratch as BitPleScratch, SolveScratch as BitSolve};
use gfm::dense::{Matrix, Ple, SmallMatrix, SolveScratch};
use gfm::hybrid::Hybrid;
use gfm::incremental::{Echelon, EchelonMode};
use gfm::{GeometryError, SolveError};

type Gf8 = FiniteField<2, 8, Polynomial<AES>>;

/// Asserts `rendered` mentions every value in `values` as a decimal number.
fn mentions(rendered: &str, values: &[usize]) {
    for value in values {
        assert!(
            rendered.contains(&value.to_string()),
            "`{value}` missing from `{rendered}`"
        );
    }
}

#[test]
fn geometry_errors_report_their_operands() {
    let overflow = Matrix::<Gf8>::zeros(usize::MAX, usize::MAX).unwrap_err();
    let GeometryError::Overflow { rows, pitch } = overflow else {
        panic!("expected an overflow, got {overflow:?}");
    };
    mentions(&overflow.to_string(), &[rows, pitch]);

    let shape = Matrix::<Gf8>::from_rows(2, 2, &[0, 1, 2]).unwrap_err();
    assert_eq!(
        shape,
        GeometryError::Shape {
            lhs: (2, 2),
            rhs: (3, 1)
        }
    );
    mentions(&shape.to_string(), &[2, 3, 1]);

    let ragged = GeometryError::Ragged {
        len: 7,
        element_bytes: 4,
    };
    mentions(&ragged.to_string(), &[7, 4]);

    #[cfg(feature = "std")]
    {
        let surfaced: &dyn std::error::Error = &shape;
        assert!(surfaced.source().is_none());
    }
}

#[test]
fn solve_errors_report_their_operands() {
    let mut hybrid = Hybrid::<Gf8>::new(2, 1);
    hybrid.push_binary_row(&[], &[1]);
    let inconsistent = hybrid.solve().unwrap_err();
    assert_eq!(inconsistent, SolveError::Inconsistent { row: 0 });
    mentions(&inconsistent.to_string(), &[0]);

    let m = Matrix::<Gf8>::from_rows(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
    let singular = Ple::decompose(m)
        .inverse_into(&mut Matrix::<Gf8>::zeros(3, 3).unwrap())
        .unwrap_err();
    assert_eq!(singular, SolveError::Singular { rank: 2, order: 3 });
    mentions(&singular.to_string(), &[2, 3]);

    #[cfg(feature = "std")]
    {
        let surfaced: &dyn std::error::Error = &singular;
        assert!(surfaced.source().is_none());
    }
}

#[test]
fn dense_debug_summaries_carry_shapes() {
    let mut m = Matrix::<Gf8>::zeros(5, 7).unwrap();
    mentions(&format!("{m:?}"), &[5, 7]);
    mentions(&format!("{:?}", m.as_view_mut()), &[5, 7]);
    let ple = Ple::decompose(Matrix::<Gf8>::identity(6).unwrap());
    mentions(&format!("{ple:?}"), &[6]);
    let small = SmallMatrix::<Gf8, 9>::zeros();
    mentions(&format!("{small:?}"), &[9]);
    assert!(!format!("{:?}", SolveScratch::<Gf8>::new()).is_empty());
}

#[test]
fn bit_debug_summaries_carry_shapes() {
    let bits = BitMatrix::zeros(5, 7).unwrap();
    mentions(&format!("{bits:?}"), &[5, 7]);
    let ple = BitPle::decompose(BitMatrix::identity(6).unwrap(), &mut BitPleScratch::new());
    mentions(&format!("{ple:?}"), &[6]);
    assert!(!format!("{:?}", BitPleScratch::new()).is_empty());
    assert!(!format!("{:?}", BitSolve::new()).is_empty());
}

#[test]
fn streaming_and_hybrid_debug_summaries_carry_shapes() {
    let mut echelon = Echelon::<Gf8>::new(5, 1, EchelonMode::PropagateUnits).unwrap();
    echelon.absorb_unit(3, &[7]);
    mentions(&format!("{echelon:?}"), &[5]);
    let retained = echelon.retained_rows().next().unwrap();
    mentions(&format!("{retained:?}"), &[3]);

    let mut hybrid = Hybrid::<Gf8>::new(4, 1);
    for column in 0..4u32 {
        hybrid.push_binary_row(&[column], &[1]);
    }
    mentions(&format!("{hybrid:?}"), &[4]);
    let solution = hybrid.solve().unwrap();
    mentions(&format!("{solution:?}"), &[4]);
}
