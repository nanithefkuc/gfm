//! Byte ingress accepts every lane the field decoder accepts, including
//! noncanonical prime-field aliases such as the modulus itself (which decodes
//! to zero). Every algebraic decision — consistency, dependence, verification,
//! and matrix equality — is taken on decoded values, so an alias behaves
//! exactly like the canonical value it names.

use fgf::prime::{Goldilocks, Mersenne31};
use fgf::{Elem, FieldBuffer, FiniteField};
use gfm::dense::{Matrix, Ple, SolveScratch};
use gfm::hybrid::Hybrid;
use gfm::incremental::{Echelon, EchelonMode, Innovation};

/// A one-byte residue row: lanes `251..=255` are aliases of `0..=4`.
type Small = FiniteField<251, 1>;

/// The lane of the modulus, which decodes to zero, and the lane of `value`.
fn alias_of_zero<F: FieldBuffer>(modulus: u64) -> Vec<u8> {
    modulus.to_le_bytes()[..F::BYTES].to_vec()
}

fn lane<F: FieldBuffer>(value: Elem<F>) -> Vec<u8> {
    let mut bytes = vec![0; F::BYTES];
    F::encode(&mut bytes, value);
    bytes
}

fn dense_zero_alias_is_consistent<F: FieldBuffer>(modulus: u64) {
    let zero = alias_of_zero::<F>(modulus);
    assert!(F::decode(&zero) == Elem::<F>::ZERO);

    // 0 · x = alias(0) is consistent, with x = 0.
    let a = Matrix::<F>::from_rows(1, 1, &lane(Elem::<F>::ZERO)).unwrap();
    let rhs = Matrix::<F>::from_rows(1, 1, &zero).unwrap();
    let ple = Ple::decompose(a);
    let mut x = Matrix::<F>::zeros(1, 1).unwrap();
    ple.solve_into(&rhs, &mut x, &mut SolveScratch::new())
        .expect("an alias of zero on the right is consistent");
    assert!(x.get(0, 0) == Elem::<F>::ZERO);

    // Equality compares values, not spellings.
    let canonical = Matrix::<F>::zeros(1, 1).unwrap();
    assert_eq!(rhs, canonical);
    let one = Matrix::<F>::from_rows(1, 1, &lane(Elem::<F>::ONE)).unwrap();
    assert_ne!(rhs, one);
}

fn echelon_zero_alias_is_dependent<F: FieldBuffer>(modulus: u64) {
    let zero = alias_of_zero::<F>(modulus);
    let mut echelon = Echelon::<F>::new(1, 1, EchelonMode::PropagateUnits).unwrap();
    assert_eq!(
        echelon.absorb(&lane(Elem::<F>::ZERO), &zero),
        Innovation::Dependent
    );
    assert_eq!(
        echelon.absorb(&lane(Elem::<F>::ZERO), &lane(Elem::<F>::ONE)),
        Innovation::Inconsistent
    );
}

fn hybrid_aliases_agree<F: FieldBuffer>(modulus: u64) {
    // x = 3 and x = 3 + modulus name the same equation; the alias still fits
    // a one-byte lane.
    let three = Elem::<F>::from_raw_u64(3);
    let alias = (modulus + 3).to_le_bytes()[..F::BYTES].to_vec();
    assert!(F::decode(&alias) == three);
    let mut hybrid = Hybrid::<F>::new(1, F::BYTES);
    hybrid.push_field_row(&[0], &[Elem::<F>::ONE], &lane(three));
    hybrid.push_field_row(&[0], &[Elem::<F>::ONE], &alias);
    let solution = hybrid.solve().expect("aliases of one value are consistent");
    assert_eq!(solution.rank(), 1);
    assert!(F::decode(solution.value(0)) == three);
}

/// `Elem::from_raw` takes each row's raw word type; the tests only need
/// small values, so route them through the byte decoder.
trait FromU64: Sized {
    fn from_raw_u64(value: u64) -> Self;
}

impl<F: FieldBuffer> FromU64 for Elem<F> {
    fn from_raw_u64(value: u64) -> Self {
        F::decode(&value.to_le_bytes()[..F::BYTES])
    }
}

#[test]
fn zero_aliases_decide_by_value() {
    dense_zero_alias_is_consistent::<Mersenne31>(0x7FFF_FFFF);
    dense_zero_alias_is_consistent::<Goldilocks>(0xFFFF_FFFF_0000_0001);
    dense_zero_alias_is_consistent::<Small>(251);
    echelon_zero_alias_is_dependent::<Mersenne31>(0x7FFF_FFFF);
    echelon_zero_alias_is_dependent::<Goldilocks>(0xFFFF_FFFF_0000_0001);
    echelon_zero_alias_is_dependent::<Small>(251);
}

#[test]
fn hybrid_verification_decides_by_value() {
    hybrid_aliases_agree::<Mersenne31>(0x7FFF_FFFF);
    hybrid_aliases_agree::<Goldilocks>(0xFFFF_FFFF_0000_0001);
    hybrid_aliases_agree::<Small>(251);
}
