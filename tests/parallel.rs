//! The Rayon row dispatcher splits long row updates across workers. Its
//! results equal the serial arithmetic for every lane width, including
//! extension lanes wider than eight bytes.

use fgf::extension::{Components, Extension, ExtensionSpec, Relation};
use fgf::prime::Goldilocks;
use fgf::{Elem, FieldBuffer, FiniteField};
use gfm::{Matrix, mul_add};

/// GF(p²) over Goldilocks as `t² = 7`; seven generates the multiplicative
/// group, so it is a non-residue and the relation is irreducible.
struct Quadratic;

impl ExtensionSpec<2> for Quadratic {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "Goldilocks[t]/(t^2 - 7)";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}

type Wide = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<Quadratic, 2>>;

/// Enough elements that one row exceeds the parallel threshold.
const COLS: usize = (4 << 20) / 16;

fn element(index: usize) -> Elem<Wide> {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&(index as u64 * 0x9E37_79B9).to_le_bytes());
    bytes[8..].copy_from_slice(&(index as u64 ^ 0x5555).to_le_bytes());
    Wide::decode(&bytes)
}

#[test]
fn wide_lanes_split_across_workers() {
    assert_eq!(Wide::BYTES, 16);
    let factor = element(12_345);
    let a = {
        let mut a = Matrix::<Wide>::zeros(1, 1).unwrap();
        a.set(0, 0, factor);
        a
    };
    let mut b = Matrix::<Wide>::zeros(1, COLS).unwrap();
    let mut c = Matrix::<Wide>::zeros(1, COLS).unwrap();
    for col in 0..COLS {
        b.set(0, col, element(col));
        c.set(0, col, element(col + 7));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| mul_add(&mut c, &a, &b));
    for col in 0..COLS {
        assert!(c.get(0, col) == element(col + 7) + factor * element(col));
    }
}
