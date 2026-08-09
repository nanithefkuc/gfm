//! The phase-defining property: `Hybrid` and a full dense `Ple` over the same
//! system agree on rank, on the solution, and on the inconsistency verdict —
//! on every generated system family and every seed. Plus the deferred
//! byte-identical / fewer-ops property, the lazy-widening count, and the
//! inactivation schedule.

#![allow(clippy::needless_range_loop)]

mod common;

use common::draw;
use fgf::field::{Elem, Field};
use fgf::{FieldKernels, Gf8, Gf16};
use gfm::{DenseRow, DenseRows, Hybrid, Matrix, Ple, PleScratch, SolveError, SolveScratch};
type Equation<F> = (Vec<u32>, Vec<<F as Field>::Elem>, Vec<u8>);

/// A generated sparse system over `n` unknowns with `sym` symbol elements per
/// row. Each row is `(support, coeffs, rhs)` with `rhs` packed bytes.
struct System<F: FieldKernels> {
    n: usize,
    sym: usize,
    rows: Vec<Equation<F>>,
}

impl<F: FieldKernels> System<F> {
    fn new(n: usize, sym: usize) -> Self {
        Self {
            n,
            sym,
            rows: Vec::new(),
        }
    }

    fn sym_len(&self) -> usize {
        self.sym * F::BYTES
    }

    /// Packs a row of symbol elements to bytes.
    fn pack(row: &[F::Elem]) -> Vec<u8> {
        let mut out = vec![0u8; row.len() * F::BYTES];
        for (i, &e) in row.iter().enumerate() {
            F::write(&mut out[i * F::BYTES..(i + 1) * F::BYTES], e);
        }
        out
    }

    /// Adds a row with the given coefficients, and a right-hand side induced
    /// by the true solution `x` (so the system stays consistent).
    fn push_consistent(&mut self, support: Vec<u32>, coeffs: Vec<F::Elem>, x: &[Vec<F::Elem>]) {
        let mut rhs = vec![F::Elem::ZERO; self.sym];
        for (&c, &co) in support.iter().zip(&coeffs) {
            for s in 0..self.sym {
                rhs[s] = rhs[s].add(co.mul(x[c as usize][s]));
            }
        }
        self.rows.push((support, coeffs, Self::pack(&rhs)));
    }

    fn build_hybrid(&self) -> Hybrid<F> {
        let mut hybrid = Hybrid::<F>::new(self.n, self.sym_len());
        hybrid.extend_rows(self);
        hybrid
    }

    fn dense(&self) -> (Matrix<F>, Matrix<F>) {
        let m = self.rows.len();
        let mut a = Matrix::<F>::zeros(m, self.n).unwrap();
        let mut b = Matrix::<F>::zeros(m, self.sym).unwrap();
        for (i, (support, coeffs, rhs)) in self.rows.iter().enumerate() {
            for (&c, &co) in support.iter().zip(coeffs) {
                a.set(i, c as usize, co);
            }
            b.row_mut(i).copy_from_slice(rhs);
        }
        (a, b)
    }

    /// Checks that `x` (per-column symbols) satisfies every equation.
    fn satisfied_by(&self, value: impl Fn(usize) -> Vec<F::Elem>) -> bool {
        let cache: Vec<Vec<F::Elem>> = (0..self.n).map(&value).collect();
        for (support, coeffs, rhs) in &self.rows {
            let mut acc = vec![F::Elem::ZERO; self.sym];
            for (&c, &co) in support.iter().zip(coeffs) {
                for s in 0..self.sym {
                    acc[s] = acc[s].add(co.mul(cache[c as usize][s]));
                }
            }
            if Self::pack(&acc) != *rhs {
                return false;
            }
        }
        true
    }
}
impl<F: FieldKernels> DenseRows<F> for System<F> {
    fn for_each_row(&self, visit: &mut impl FnMut(DenseRow<'_, F>)) {
        for (support, coeffs, rhs) in &self.rows {
            if coeffs.iter().all(|coefficient| coefficient.is_one()) {
                visit(DenseRow::Binary { support, rhs });
            } else {
                visit(DenseRow::Field {
                    support,
                    coeffs,
                    rhs,
                });
            }
        }
    }
}

/// A random true solution.
fn random_solution<F: FieldKernels>(n: usize, sym: usize, seed: u64) -> Vec<Vec<F::Elem>> {
    let mut st = seed | 1;
    (0..n)
        .map(|_| {
            (0..sym)
                .map(|_| {
                    let mut bytes = [0u8; 8];
                    for b in bytes.iter_mut().take(F::BYTES) {
                        *b = draw(&mut st, 256) as u8;
                    }
                    F::read(&bytes[..F::BYTES])
                })
                .collect()
        })
        .collect()
}

/// A sorted, distinct random support of the given weight over `0..n`.
fn support(n: usize, weight: usize, st: &mut u64) -> Vec<u32> {
    let mut cols = Vec::new();
    let mut guard = 0;
    while cols.len() < weight && guard < weight * 20 {
        let c = draw(st, n) as u32;
        if !cols.contains(&c) {
            cols.push(c);
        }
        guard += 1;
    }
    cols.sort_unstable();
    cols
}
/// Dense-`Ple` reference: rank, consistency, solution, and the columns whose
/// values are invariant across every solution.
fn ple_reference<F: FieldKernels>(sys: &System<F>) -> (usize, bool, Option<Matrix<F>>, Vec<bool>) {
    let (a, b) = sys.dense();
    let ple = Ple::decompose(a, &mut PleScratch::new());
    let rank = ple.rank();
    let mut kernel = Matrix::<F>::zeros(sys.n, sys.n - rank).unwrap();
    ple.kernel_into(&mut kernel);
    let determined = (0..sys.n)
        .map(|row| kernel.row(row).iter().all(|&byte| byte == 0))
        .collect();
    let mut out = Matrix::<F>::zeros(sys.n, sys.sym).unwrap();
    match ple.solve_into(&b, &mut out, &mut SolveScratch::new()) {
        Ok(()) => (rank, true, Some(out), determined),
        Err(SolveError::Inconsistent { .. }) => (rank, false, None, determined),
        Err(e) => panic!("unexpected solve error: {e:?}"),
    }
}

/// The whole property, on one system.
fn assert_matches_ple<F: FieldKernels>(sys: &System<F>) {
    let (rank_p, consistent_p, sol_p, determined_p) = ple_reference(sys);
    let mut hybrid = sys.build_hybrid();
    match hybrid.solve() {
        Ok(sol) => {
            assert!(consistent_p, "hybrid solved an inconsistent system");
            assert_eq!(sol.rank(), rank_p, "rank");
            // The hybrid solution satisfies every original equation.
            assert!(
                sys.satisfied_by(|c| read_row::<F>(sol.value(c), sys.sym)),
                "hybrid solution does not satisfy the system"
            );
            let p = sol_p.unwrap();
            assert!(
                sys.satisfied_by(|column| read_row::<F>(p.row(column), sys.sym)),
                "dense Ple solution does not satisfy the {}x{} rank-{} system",
                sys.rows.len(),
                sys.n,
                rank_p
            );
            for (column, &expected_determined) in determined_p.iter().enumerate() {
                assert_eq!(
                    sol.is_determined(column),
                    expected_determined,
                    "determinedness mismatch at col {column}"
                );
                if expected_determined {
                    assert_eq!(
                        sol.value(column),
                        p.row(column),
                        "determined solution mismatch at col {column}"
                    );
                }
            }
        }
        Err(SolveError::Inconsistent { .. }) => {
            assert!(!consistent_p, "hybrid rejected a consistent system");
        }
        Err(e) => panic!("unexpected hybrid error: {e:?}"),
    }
}

fn read_row<F: FieldKernels>(bytes: &[u8], sym: usize) -> Vec<F::Elem> {
    (0..sym)
        .map(|s| F::read(&bytes[s * F::BYTES..(s + 1) * F::BYTES]))
        .collect()
}

#[test]
fn random_sparse_matches_ple() {
    for seed in 0..40u64 {
        let mut st = 0x5A5A_0000 ^ seed;
        let n = 8 + draw(&mut st, 40);
        let m = n + draw(&mut st, 20); // over-determined-ish
        let sym = 1 + draw(&mut st, 3);
        let x = random_solution::<Gf8>(n, sym, seed ^ 0x1234);
        let mut sys = System::<Gf8>::new(n, sym);
        for _ in 0..m {
            let w = 1 + draw(&mut st, 5);
            let sup = support(n, w, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        assert_matches_ple(&sys);
    }
}

#[test]
fn ldpc_shaped_matches_ple() {
    // Regular low-density: fixed small row weight, enough rows for full rank.
    for seed in 0..20u64 {
        let mut st = 0x1D0C_0000 ^ seed;
        let n = 20 + draw(&mut st, 40);
        let sym = 2;
        let x = random_solution::<Gf8>(n, sym, seed);
        let mut sys = System::<Gf8>::new(n, sym);
        for _ in 0..(n + n / 2) {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        assert_matches_ple(&sys);
    }
}

#[test]
fn stopping_set_shaped_matches_ple() {
    // An adversarial cluster S: every row that touches S touches >= 2 of it,
    // so weight-1 peeling stalls on S and the schedule must inactivate.
    for seed in 0..20u64 {
        let mut st = 0x5700_0000u64.wrapping_add(seed);
        let n = 24 + draw(&mut st, 20);
        let sym = 2;
        let s_size = 4 + draw(&mut st, 4);
        let s: Vec<u32> = (0..s_size as u32).collect();
        let x = random_solution::<Gf8>(n, sym, seed);
        let mut sys = System::<Gf8>::new(n, sym);
        // Rows over S (each two S-columns plus an outside column).
        for _ in 0..(n) {
            let a = s[draw(&mut st, s.len())];
            let mut b = s[draw(&mut st, s.len())];
            if b == a {
                b = s[(a as usize + 1) % s.len()];
            }
            let outside = (s_size + draw(&mut st, n - s_size)) as u32;
            let mut sup = vec![a, b, outside];
            sup.sort_unstable();
            sup.dedup();
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        // Plenty of light rows elsewhere for overall solvability.
        for _ in 0..n {
            let sup = support(n, 2, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        assert_matches_ple(&sys);
    }
}

#[test]
fn field_band_matches_ple() {
    // Binary sparse rows plus a band of dense GF(2^8) rows (the HDPC shape).
    for seed in 0..20u64 {
        let mut st = 0xF1E1_0000 ^ seed;
        let n = 16 + draw(&mut st, 24);
        let sym = 2;
        let x = random_solution::<Gf8>(n, sym, seed);
        let mut sys = System::<Gf8>::new(n, sym);
        for _ in 0..n {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        // A few dense field rows over all columns.
        for _ in 0..6 {
            let sup: Vec<u32> = (0..n as u32).collect();
            let coeffs: Vec<_> = (0..n)
                .map(|_| {
                    let v = 1 + draw(&mut st, 255);
                    <Gf8 as Field>::read(&[v as u8])
                })
                .collect();
            sys.push_consistent(sup, coeffs, &x);
        }
        assert_matches_ple(&sys);
    }
}

#[cfg(feature = "internals")]
#[test]
fn compact_dispatch_covers_max_order() {
    // A = J + 3I at odd order 65 is nonsingular in characteristic two:
    // J is one on the all-ones vector and zero on the sum-zero subspace,
    // while 3I shifts both eigenvalues away from zero. Every coefficient is
    // nonzero, so peeling inactivates 64 columns before the final pivot.
    let n = 65;
    let sym = 2;
    let x = random_solution::<Gf8>(n, sym, 0x64_C0FF);
    let support: Vec<u32> = (0..n as u32).collect();
    let one = <Gf8 as Field>::Elem::ONE;
    let diagonal = Gf8::read(&[2]);
    let mut sys = System::<Gf8>::new(n, sym);
    for row in 0..n {
        let mut coefficients = vec![one; n];
        coefficients[row] = diagonal;
        sys.push_consistent(support.clone(), coefficients, &x);
    }

    let (solution, stats) = sys.build_hybrid().solve_with_stats(true).unwrap();
    assert!(solution.is_full_rank());
    assert_eq!(stats.inactivations, 64);
    assert!(sys.satisfied_by(|column| read_row::<Gf8>(solution.value(column), sym)));
}

#[test]
fn inconsistent_systems_are_rejected() {
    for seed in 0..20u64 {
        let mut st = 0x1FC0_0000 ^ seed;
        let n = 10 + draw(&mut st, 20);
        let sym = 2;
        let x = random_solution::<Gf8>(n, sym, seed);
        let mut sys = System::<Gf8>::new(n, sym);
        for _ in 0..(n + 5) {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8 as Field>::Elem::ONE; sup.len()], &x);
        }
        // Duplicate an existing row's coefficients with a corrupted RHS.
        let (sup, co, mut rhs) = sys.rows[0].clone();
        rhs[0] ^= 0xFF;
        sys.rows.push((sup, co, rhs));
        assert_matches_ple(&sys);
    }
}

#[test]
fn wider_field_matches_ple() {
    for seed in 0..15u64 {
        let mut st = 0x16A0_0000 ^ seed;
        let n = 10 + draw(&mut st, 20);
        let sym = 2;
        let x = random_solution::<Gf16>(n, sym, seed);
        let mut sys = System::<Gf16>::new(n, sym);
        for _ in 0..(n + 8) {
            let w = 1 + draw(&mut st, 4);
            let sup = support(n, w, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf16 as Field>::Elem::ONE; sup.len()], &x);
        }
        assert_matches_ple(&sys);
    }
}

#[cfg(feature = "internals")]
fn assert_same_solution<F: FieldKernels>(
    left: &gfm::Solution<F>,
    right: &gfm::Solution<F>,
    n: usize,
) {
    assert_eq!(left.rank(), right.rank());
    for column in 0..n {
        assert_eq!(left.is_determined(column), right.is_determined(column));
        assert_eq!(left.value(column), right.value(column));
    }
}

#[cfg(feature = "internals")]
#[test]
fn deferred_is_byte_identical_and_uses_fewer_row_ops() {
    let n = 32;
    let sym = 64;
    let x = random_solution::<Gf8>(n, sym, 0xD3FE_0001);
    let mut sys = System::<Gf8>::new(n, sym);
    for column in 0..n {
        let support = vec![column as u32];
        sys.push_consistent(support.clone(), vec![<Gf8 as Field>::Elem::ONE], &x);
    }
    // Redundant received equations: eager application updates each duplicate;
    // deferred replay omits them from the independent dense/pivot set.
    for repeat in 0..4 {
        for column in 0..n {
            let other = (column + repeat + 1) % n;
            let mut support = vec![column as u32, other as u32];
            support.sort_unstable();
            sys.push_consistent(support, vec![<Gf8 as Field>::Elem::ONE; 2], &x);
        }
    }

    let (eager_solution, eager) = sys.build_hybrid().solve_with_stats(false).unwrap();
    let (deferred_solution, deferred) = sys.build_hybrid().solve_with_stats(true).unwrap();
    assert_same_solution(&eager_solution, &deferred_solution, n);
    assert!(
        deferred.row_ops < eager.row_ops,
        "deferred={} eager={}",
        deferred.row_ops,
        eager.row_ops
    );
}

#[cfg(feature = "internals")]
#[test]
fn field_band_widens_exactly_the_rows_it_touches() {
    let band = 8;
    let n = band * 2;
    let sym = 4;
    let x = random_solution::<Gf8>(n, sym, 0xBADD_0001);
    let mut sys = System::<Gf8>::new(n, sym);
    let alpha = <Gf8 as Field>::read(&[2]);

    // All field pivots precede their binary targets. Each pivot touches one
    // distinct binary row, so exactly `band` transitions are necessary.
    for index in 0..band {
        sys.push_consistent(vec![(2 * index) as u32], vec![alpha], &x);
    }
    for index in 0..band {
        sys.push_consistent(
            vec![(2 * index) as u32, (2 * index + 1) as u32],
            vec![<Gf8 as Field>::Elem::ONE; 2],
            &x,
        );
    }

    let (solution, stats) = sys.build_hybrid().solve_with_stats(true).unwrap();
    assert!(solution.is_full_rank());
    assert_eq!(stats.widenings, band);
}

#[cfg(feature = "internals")]
const RFC_DEGREE_THRESHOLDS: [u32; 31] = [
    0, 5_243, 529_531, 704_294, 791_675, 844_104, 879_057, 904_023, 922_747, 937_311, 948_962,
    958_494, 966_438, 973_160, 978_921, 983_914, 988_283, 992_138, 995_565, 998_631, 1_001_391,
    1_003_887, 1_006_157, 1_008_229, 1_010_129, 1_011_876, 1_013_490, 1_014_983, 1_016_370,
    1_017_662, 1_048_576,
];

#[cfg(feature = "internals")]
fn rfc_degree(value: usize, columns: usize) -> usize {
    let degree = RFC_DEGREE_THRESHOLDS.partition_point(|&threshold| threshold <= value as u32);
    degree.min(columns.saturating_sub(2).max(1))
}

#[cfg(feature = "internals")]
fn rfc_degree_system(columns: usize, seed: u64) -> System<Gf8> {
    let mut state = seed;
    let overhead = (columns as f64).sqrt().ceil() as usize + 8;
    let mut system = System::<Gf8>::new(columns, 1);
    for _ in 0..(columns + overhead) {
        let degree = rfc_degree(draw(&mut state, 1 << 20), columns);
        let support = support(columns, degree, &mut state);
        system.rows.push((
            support.clone(),
            vec![<Gf8 as Field>::Elem::ONE; support.len()],
            vec![0],
        ));
    }
    system
}

#[cfg(feature = "internals")]
#[test]
fn rfc_degree_inactivation_scales_with_square_root() {
    let sizes = [10, 25, 50, 100, 250, 500, 1_000];
    let mut worst = 0.0f64;
    for &columns in &sizes {
        let mut maximum = 0;
        for seed in 0..8 {
            let (_, stats) = rfc_degree_system(columns, seed)
                .build_hybrid()
                .solve_with_stats(true)
                .unwrap();
            maximum = maximum.max(stats.inactivations);
        }
        let ratio = maximum as f64 / (columns as f64).sqrt();
        worst = worst.max(ratio);
        eprintln!("k={columns} max_g={maximum} g/sqrt(k)={ratio:.3}");
    }
    assert!(worst <= 3.0, "g/sqrt(k) grew to {worst:.3}");
}
