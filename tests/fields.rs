//! The dense domain over field presentations `fgf` 4 makes reachable beyond
//! the named binary towers and prime fields: generic prime residues at every
//! lane width, the eight-byte GF(2) residue row, ordered-basis GF(2^8)
//! presentations whose one is not the integer one, and custom extensions
//! with dense defining relations. Each field runs the same algebraic
//! certificates against the naive oracles.

mod common;
mod oracles;

use fgf::binary::{AES, Cantor, Normal};
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};
use fgf::prime::Residue;
use fgf::{Elem, FieldBuffer, FieldElem, FiniteField};
use gfm::dense::{Matrix, Ple, SolveScratch, mul_into};
use oracles::{Naive, crate_matrix, naive_identity, naive_mul, naive_of, naive_with_rank};

type Residue8 = FiniteField<251, 1>;
type Residue16 = FiniteField<65_521, 1>;
type Residue32 = FiniteField<4_294_967_291, 1>;
type Residue64 = FiniteField<18_446_744_073_709_551_557, 1>;
type ResidueGf2 = FiniteField<2, 1>;
type NormalGf8 = FiniteField<2, 8, Normal<AES, 0x20>>;
type CantorGf8 = FiniteField<2, 8, Cantor<AES, 0x20>>;

/// GF(3^40) by a dense irreducible relation over the one-byte GF(3) residue
/// row: forty-byte lanes.
struct Gf3Dense40Spec;
impl ExtensionSpec<40> for Gf3Dense40Spec {
    type Base = FiniteField<3, 1, Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(3^40) dense";
    const RELATION: Relation = Relation::from_coordinates(&[
        2, 0, 2, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 1, 1,
        0, 2, 0, 0, 2, 1, 0, 1, 2, 1,
    ]);
}
type Gf3Dense40 = FiniteField<3, 40, Extension<Gf3Dense40Spec, 40>>;

/// GF(9) as GF(3)[t]/(t² + 1), then a dense relative-degree-eight relation
/// over it: GF(3^16) with sixteen-byte lanes over a non-prime base.
struct Gf9Spec;
impl ExtensionSpec<2> for Gf9Spec {
    type Base = FiniteField<3, 1, Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(9)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type Gf9 = FiniteField<3, 2, Extension<Gf9Spec, 2>>;

struct Gf9Dense8Spec;
impl ExtensionSpec<8> for Gf9Dense8Spec {
    type Base = Gf9;
    type Storage = Components;
    const NAME: &'static str = "GF(3^16) dense";
    const RELATION: Relation =
        Relation::from_coordinates(&[1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0, 0, 0, 2]);
}
type Gf9Dense8 = FiniteField<3, 16, Extension<Gf9Dense8Spec, 8>>;

/// Inverse, solve, rank, and right kernel over `F`, each checked against the
/// naive oracles rather than against another `gfm` path.
fn certify<F: FieldBuffer>(seed: u64) {
    let n = 5;
    // Full rank: A · A⁻¹ = I.
    let a = naive_with_rank::<F>(n, n, n, seed);
    let ple = Ple::decompose(crate_matrix(&a));
    assert_eq!(ple.rank(), n, "{} full rank", F::NAME);
    let mut inverse = Matrix::<F>::zeros(n, n).unwrap();
    ple.inverse_into(&mut inverse).unwrap();
    assert!(
        naive_mul(&a, &naive_of(&inverse)) == naive_identity::<F>(n),
        "{} inverse",
        F::NAME
    );

    // Solve: A · x = b for b = A · x₀.
    let x0 = naive_with_rank::<F>(n, 2, 2, seed ^ 0x55);
    let b = naive_mul(&a, &x0);
    let mut x = Matrix::<F>::zeros(n, 2).unwrap();
    ple.solve_into(&crate_matrix(&b), &mut x, &mut SolveScratch::new())
        .unwrap();
    assert!(naive_of(&x) == x0, "{} solve", F::NAME);

    // Rank deficiency: the right kernel has n − rank columns and A·K = 0.
    let rank = 3;
    let d = naive_with_rank::<F>(n, n + 1, rank, seed ^ 0xAA);
    let deficient = Ple::decompose(crate_matrix(&d));
    assert_eq!(deficient.rank(), rank, "{} deficient rank", F::NAME);
    let mut kernel = Matrix::<F>::zeros(n + 1, n + 1 - rank).unwrap();
    deficient.right_kernel_into(&mut kernel);
    let product: Naive<F> = naive_mul(&d, &naive_of(&kernel));
    assert!(
        product.iter().flatten().all(|e| e.is_zero()),
        "{} kernel",
        F::NAME
    );
    let mut product = Matrix::<F>::zeros(n, n + 1 - rank).unwrap();
    mul_into(&mut product, &crate_matrix(&d), &kernel);
    assert!(product == Matrix::<F>::zeros(n, n + 1 - rank).unwrap());

    // The identity is the field's one, whatever its raw coordinates.
    let identity = Matrix::<F>::identity(2).unwrap();
    assert!(identity.get(0, 0) == Elem::<F>::ONE);
    assert!(identity.get(0, 1) == Elem::<F>::ZERO);
}

#[test]
fn residue_rows_at_every_lane_width() {
    certify::<Residue8>(1);
    certify::<Residue16>(2);
    certify::<Residue32>(3);
    certify::<Residue64>(4);
    certify::<ResidueGf2>(5);
}

#[test]
fn ordered_basis_presentations() {
    assert_ne!(Elem::<NormalGf8>::ONE.to_raw(), 1);
    certify::<NormalGf8>(6);
    certify::<CantorGf8>(7);
}

#[test]
fn dense_relation_extensions() {
    assert_eq!(Gf3Dense40::BYTES, 40);
    assert_eq!(Gf9Dense8::BYTES, 16);
    certify::<Gf3Dense40>(8);
    certify::<Gf9Dense8>(9);
}
