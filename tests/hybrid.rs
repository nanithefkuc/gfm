//! `Hybrid` and a full dense `Ple` over the same system agree on rank, on the
//! solution, and on the inconsistency verdict — across generated system
//! families, hand-built elimination shapes, initially inactive column sets,
//! deferred rows, the hot GF(2) merge paths, and payload-only replay. The
//! schedule is checked alongside the answer: deferral stays byte-identical
//! while doing fewer row operations, widening touches exactly the rows it
//! must, and inactivation stays at the square-root scale.

// Index arithmetic across several matrices at once; iterators would obscure it.
#![allow(clippy::needless_range_loop)]

mod common;

use common::draw;
use fgf::field::{Elem, Field};
use fgf::{FieldKernels, Gf8B, Gf16};
use gfm::internals::HybridInternals;
use gfm::{DenseRow, DenseRows, Hybrid, Matrix, Ple, PleScratch, SolveError, SolveScratch};

type Equation<F> = (Vec<u32>, Vec<<F as Field>::Elem>, Vec<u8>);
type E = <Gf8B as Field>::Elem;

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
            F::encode(&mut out[i * F::BYTES..(i + 1) * F::BYTES], e);
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
        self.build_hybrid_with(&[])
    }

    /// Builds the solver with `initial_inactive` pre-inactivated.
    fn build_hybrid_with(&self, initial_inactive: &[u32]) -> Hybrid<F> {
        let mut hybrid =
            Hybrid::<F>::with_initial_inactive(self.n, self.sym_len(), initial_inactive);
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
                    F::decode(&bytes[..F::BYTES])
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

/// The packed payload induced by `support`/`coeffs` at the true solution `x`,
/// for tests that push rows straight into a `Hybrid`.
fn payload(support: &[u32], coeffs: &[E], x: &[Vec<E>], sym: usize) -> Vec<u8> {
    let mut rhs = vec![E::ZERO; sym];
    for (&c, &co) in support.iter().zip(coeffs) {
        for s in 0..sym {
            rhs[s] = rhs[s].add(co.mul(x[c as usize][s]));
        }
    }
    System::<Gf8B>::pack(&rhs)
}

/// The same payload for a support whose coefficients are all one.
fn unit_payload(support: &[u32], x: &[Vec<E>], sym: usize) -> Vec<u8> {
    let coeffs = vec![E::ONE; support.len()];
    payload(support, &coeffs, x, sym)
}

/// The oracle system behind explicit `(support, coeffs)` rows: the same
/// equations, with right-hand sides induced by the true solution `x`.
fn system_of(n: usize, sym: usize, rows: &[(Vec<u32>, Vec<E>)], x: &[Vec<E>]) -> System<Gf8B> {
    let mut sys = System::<Gf8B>::new(n, sym);
    for (support, coeffs) in rows {
        sys.push_consistent(support.clone(), coeffs.clone(), x);
    }
    sys
}

/// The same, for supports whose coefficients are all one.
fn binary_system_of(n: usize, sym: usize, supports: &[Vec<u32>], x: &[Vec<E>]) -> System<Gf8B> {
    let rows: Vec<(Vec<u32>, Vec<E>)> = supports
        .iter()
        .map(|support| (support.clone(), vec![E::ONE; support.len()]))
        .collect();
    system_of(n, sym, &rows, x)
}

fn read_row<F: FieldKernels>(bytes: &[u8], sym: usize) -> Vec<F::Elem> {
    (0..sym)
        .map(|s| F::decode(&bytes[s * F::BYTES..(s + 1) * F::BYTES]))
        .collect()
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

/// The whole property, on one already-solved system: the inconsistency
/// verdict, the rank, that the answer satisfies every original equation,
/// determinedness column by column, the value of every determined column —
/// and, when the planted solution is supplied, that every determined column
/// reads that planted value back exactly.
fn assert_solution_matches_ple<F: FieldKernels>(
    sys: &System<F>,
    solved: &Result<gfm::Solution<F>, SolveError>,
    planted: Option<&[Vec<F::Elem>]>,
) {
    let (rank_p, consistent_p, sol_p, determined_p) = ple_reference(sys);
    match solved {
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
                    if let Some(x) = planted {
                        let packed = System::<F>::pack(&x[column]);
                        assert_eq!(
                            sol.value(column),
                            &packed[..],
                            "determined column {column} does not recover the planted solution"
                        );
                    }
                }
            }
        }
        Err(SolveError::Inconsistent { .. }) => {
            assert!(!consistent_p, "hybrid rejected a consistent system");
        }
        Err(e) => panic!("unexpected hybrid error: {e:?}"),
    }
}

/// The whole property, on one system.
fn assert_matches_ple<F: FieldKernels>(sys: &System<F>) {
    assert_matches_ple_with(sys, &[])
}

/// The whole property, on one system whose `initial_inactive` columns start
/// in the inactive set.
fn assert_matches_ple_with<F: FieldKernels>(sys: &System<F>, initial_inactive: &[u32]) {
    let mut hybrid = sys.build_hybrid_with(initial_inactive);
    let solved = hybrid.solve();
    assert_solution_matches_ple(sys, &solved, None);
}

#[test]
fn random_sparse_matches_ple() {
    for seed in 0..40u64 {
        let mut st = 0x5A5A_0000 ^ seed;
        let n = 8 + draw(&mut st, 40);
        let m = n + draw(&mut st, 20); // over-determined-ish
        let sym = 1 + draw(&mut st, 3);
        let x = random_solution::<Gf8B>(n, sym, seed ^ 0x1234);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..m {
            let w = 1 + draw(&mut st, 5);
            let sup = support(n, w, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
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
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..(n + n / 2) {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
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
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
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
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
        }
        // Plenty of light rows elsewhere for overall solvability.
        for _ in 0..n {
            let sup = support(n, 2, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
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
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..n {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
        }
        // A few dense field rows over all columns.
        for _ in 0..6 {
            let sup: Vec<u32> = (0..n as u32).collect();
            let coeffs: Vec<_> = (0..n)
                .map(|_| {
                    let v = 1 + draw(&mut st, 255);
                    <Gf8B as Field>::decode(&[v as u8])
                })
                .collect();
            sys.push_consistent(sup, coeffs, &x);
        }
        assert_matches_ple(&sys);
    }
}

#[test]
fn compact_dispatch_covers_max_order() {
    // A = J + 3I at odd order 65 is nonsingular in characteristic two:
    // J is one on the all-ones vector and zero on the sum-zero subspace,
    // while 3I shifts both eigenvalues away from zero. Every coefficient is
    // nonzero, so peeling inactivates 64 columns before the final pivot.
    let n = 65;
    let sym = 2;
    let x = random_solution::<Gf8B>(n, sym, 0x64_C0FF);
    let support: Vec<u32> = (0..n as u32).collect();
    let one = <Gf8B as Field>::Elem::ONE;
    let diagonal = Gf8B::decode(&[2]);
    let mut sys = System::<Gf8B>::new(n, sym);
    for row in 0..n {
        let mut coefficients = vec![one; n];
        coefficients[row] = diagonal;
        sys.push_consistent(support.clone(), coefficients, &x);
    }

    let (solution, stats) = sys.build_hybrid().solve_reported(true).unwrap();
    assert!(solution.is_full_rank());
    assert_eq!(stats.inactivations, 64);
    assert!(sys.satisfied_by(|column| read_row::<Gf8B>(solution.value(column), sym)));
}

#[test]
fn inconsistent_systems_are_rejected() {
    for seed in 0..20u64 {
        let mut st = 0x1FC0_0000 ^ seed;
        let n = 10 + draw(&mut st, 20);
        let sym = 2;
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..(n + 5) {
            let sup = support(n, 3, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
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

#[test]
fn identity_system_solves_exactly() {
    let (n, sym) = (5usize, 3usize);
    let x = random_solution::<Gf8B>(n, sym, 0x1D);
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for c in 0..n {
        hybrid.push_binary_row(&[c as u32], &payload(&[c as u32], &[E::ONE], &x, sym));
    }
    let solved = hybrid.solve().unwrap();
    assert_eq!(solved.rank(), n);
    assert!(solved.is_full_rank());
    for c in 0..n {
        assert!(solved.is_determined(c));
        let mut expected = vec![0u8; sym];
        for s in 0..sym {
            Gf8B::encode(&mut expected[s..=s], x[c][s]);
        }
        assert_eq!(solved.value(c), &expected[..]);
    }
}

/// Duplicate equations force cancelled pivot probes: the column index still
/// lists rows whose entries already cancelled, and those probes change
/// nothing observable.
#[test]
fn duplicate_rows_leave_rank_solution_and_determinedness() {
    let (n, sym) = (24usize, 3usize);
    let x = random_solution::<Gf8B>(n, sym, 0xD401);
    let mut st = 0xD402u64;
    let mut rows: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for _ in 0..(n + 10) {
        let w = 1 + draw(&mut st, 4);
        let mut cols = Vec::new();
        while cols.len() < w {
            let c = draw(&mut st, n) as u32;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        cols.sort_unstable();
        rows.push((cols.clone(), vec![E::ONE; cols.len()]));
    }
    // Duplicate the first six rows verbatim: every duplicate's entries
    // cancel against the original during elimination.
    for i in 0..6 {
        rows.push(rows[i].clone());
    }
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, coeffs) in &rows {
        hybrid.push_binary_row(support, &payload(support, coeffs, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
}

/// Field rows whose supports nest force merges past every input weight and
/// cancellation probes in the frozen migration.
#[test]
fn nested_field_rows_match_ple() {
    let (n, sym) = (20usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0xD411);
    let alpha = Gf8B::decode(&[0x1B]);
    let beta = Gf8B::decode(&[0x36]);
    // Nested supports: each row contains the previous one plus a fresh
    // column, so merges widen past every input weight and later probes
    // revisit cancelled entries.
    let mut rows: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for k in 2..=10 {
        let support: Vec<u32> = (0..k as u32).collect();
        let coeffs: Vec<E> = (0..k)
            .map(|i| if i % 2 == 0 { alpha } else { beta })
            .collect();
        rows.push((support, coeffs));
    }
    for c in 0..n {
        rows.push((vec![c as u32], vec![E::ONE]));
    }
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, coeffs) in &rows {
        hybrid.push_field_row(support, coeffs, &payload(support, coeffs, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
}

/// One dense field row sharing its columns with many binary rows: the same
/// column is probed after its entries moved to frozen words.
#[test]
fn dense_row_over_binaries_matches_ple() {
    let (n, sym) = (16usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0xD421);
    let mut st = 0xD422u64;
    let mut rows: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for _ in 0..(n + 4) {
        let w = 2 + draw(&mut st, 3);
        let mut cols = Vec::new();
        while cols.len() < w {
            let c = draw(&mut st, n) as u32;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        cols.sort_unstable();
        rows.push((cols.clone(), vec![E::ONE; cols.len()]));
    }
    // A full-width field row: after inactivation its entries migrate to
    // frozen words, and later probes revisit them.
    let support: Vec<u32> = (0..n as u32).collect();
    let coeffs: Vec<E> = (0..n)
        .map(|i| Gf8B::decode(&[0x01 + (i as u8 % 0x7F) + 1]))
        .collect();
    rows.push((support, coeffs));
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, coeffs) in &rows {
        if coeffs.iter().all(|c| c.is_one()) {
            hybrid.push_binary_row(support, &payload(support, coeffs, &x, sym));
        } else {
            hybrid.push_field_row(support, coeffs, &payload(support, coeffs, &x, sym));
        }
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
}

#[test]
fn field_rows_with_distant_inactivation_match_ple() {
    // Two field rows on disjoint pairs plus identities elsewhere: the first
    // weight-2 edge inactivates a column the other field row never touches,
    // so its queue entry is re-parked at the same weight.
    let (n, sym) = (8usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x4C);
    let (alpha, beta) = (Gf8B::decode(&[0x1B]), Gf8B::decode(&[0x36]));
    let (gamma, delta) = (Gf8B::decode(&[0x5D]), Gf8B::decode(&[0xA6]));
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    let rows: Vec<(Vec<u32>, Vec<E>)> = vec![
        (vec![0, 1], vec![alpha, beta]),
        (vec![2, 3], vec![gamma, delta]),
        (vec![0, 1], vec![beta, alpha]),
        (vec![2, 3], vec![delta, gamma]),
        (vec![4], vec![E::ONE]),
        (vec![5], vec![E::ONE]),
        (vec![6], vec![E::ONE]),
        (vec![7], vec![E::ONE]),
    ];
    for (support, coeffs) in &rows {
        hybrid.push_field_row(support, coeffs, &payload(support, coeffs, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
    assert!(solved.unwrap().is_full_rank());
}

#[test]
fn field_pivot_with_frozen_entries_matches_ple() {
    // A field pivot carrying far inactive columns merges them into a wide
    // binary destination through the cold list-merge path.
    let (n, sym) = (12usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x6E);
    let initial: Vec<u32> = vec![7, 8, 9];
    let mut hybrid = Hybrid::<Gf8B>::with_initial_inactive(n, sym, &initial);
    let alpha = Gf8B::decode(&[0x1B]);
    // Weight-1 active support over column 0 with three frozen entries: it
    // pivots first and its snapshot spans five columns.
    hybrid.push_field_row(
        &[0, 7, 8, 9],
        &[E::ONE, E::ONE, alpha, E::ONE],
        &payload(&[0, 7, 8, 9], &[E::ONE, E::ONE, alpha, E::ONE], &x, sym),
    );
    hybrid.push_binary_row(
        &[0, 1, 2, 3],
        &payload(&[0, 1, 2, 3], &[E::ONE; 4], &x, sym),
    );
    for c in 0..n {
        hybrid.push_binary_row(&[c as u32], &payload(&[c as u32], &[E::ONE], &x, sym));
    }
    let solved = hybrid.solve().unwrap();
    assert!(solved.is_full_rank());
    for c in 0..n {
        let mut expected = vec![0u8; sym];
        for s in 0..sym {
            Gf8B::encode(&mut expected[s..=s], x[c][s]);
        }
        assert_eq!(solved.value(c), &expected[..], "column {c}");
    }
}

/// A rank-deficient band: the dense block is singular, so the kernel lift
/// runs and undetermined columns read zero.
#[test]
fn rank_deficient_band_reports_rank_and_zero_free_columns() {
    let (n, sym) = (14usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0xD431);
    // Only columns 0..8 ever appear: rank is at most 8 and columns 8..14
    // are free.
    let mut st = 0xD432u64;
    let mut rows: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for _ in 0..20 {
        let w = 1 + draw(&mut st, 4);
        let mut cols = Vec::new();
        while cols.len() < w {
            let c = draw(&mut st, 8) as u32;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        cols.sort_unstable();
        rows.push((cols.clone(), vec![E::ONE; cols.len()]));
    }
    let sys = system_of(n, sym, &rows, &x);
    let (rank, _, _, determined) = ple_reference(&sys);
    assert!(rank < n, "test needs a deficient system, got rank {rank}");
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, coeffs) in &rows {
        hybrid.push_binary_row(support, &payload(support, coeffs, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&sys, &solved, Some(&x));
    let solved = solved.unwrap();
    assert!(!solved.is_full_rank());
    for c in 0..n {
        if !determined[c] {
            assert!(
                solved.value(c).iter().all(|&b| b == 0),
                "free column {c} is not zero"
            );
        }
    }
}

/// The hot GF(2) merge: binary-on-binary elimination that cannot take the
/// fused singleton path. A weight-two system has no weight-one row for the
/// scheduler to peel, so it must inactivate a column before it can pivot,
/// and the resulting elimination goes through the merge (`axpy_xor_parts`
/// in `hybrid/sparse.rs`, called from `hybrid/solve.rs`) instead of the
/// fused singleton branch, which needs a pivot whose active list is the
/// pivot column alone. The supports are drawn so overlaps both cancel — a
/// shared column drops out of the XOR — and widen — a column the
/// destination never carried enters it and must be indexed for future
/// pivots. The observable claim: the system solves to the same rank,
/// determinedness, and determined values as a dense `Ple`, and the
/// determined columns recover the planted solution.
#[test]
fn weight_two_systems_eliminate_through_the_hot_binary_merge() {
    let (n, sym) = (16usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x60A1);
    let supports: Vec<Vec<u32>> = vec![
        // Three redundant full covers of the column set: each cover sums
        // to one equation per column, and their union keeps every column
        // degree high enough that no inactivation strands a weight-one row.
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![6, 7],
        vec![8, 9],
        vec![10, 11],
        vec![12, 13],
        vec![14, 15],
        vec![0, 2],
        vec![4, 6],
        vec![8, 10],
        vec![12, 14],
        vec![1, 3],
        vec![5, 7],
        vec![9, 11],
        vec![13, 15],
        // Cross-links: each shares one column with a pair above (cancels
        // in the merge) and pairs it with a distant one (widens it).
        vec![0, 8],
        vec![1, 9],
        vec![2, 10],
        vec![3, 11],
        vec![4, 12],
        vec![5, 13],
        vec![6, 14],
        vec![7, 15],
        vec![0, 4],
        vec![8, 12],
        vec![1, 5],
        vec![9, 13],
    ];
    for support in &supports {
        assert_eq!(support.len(), 2);
    }
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for support in &supports {
        hybrid.push_binary_row(support, &unit_payload(support, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&binary_system_of(n, sym, &supports, &x), &solved, Some(&x));
}

/// The two residual `continue` arms around the merge call site are live
/// elimination facts, not dead code: a duplicate equation leaves a stale
/// column-index listing whose entry already cancelled, and a singleton
/// pivot still probes rows that never carried its column. Both probes
/// change nothing observable — rank, solution, and determinedness still
/// match the dense oracle.
#[test]
fn stale_and_absent_pivot_probes_change_nothing() {
    let (n, sym) = (16usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x60B2);
    let mut supports: Vec<Vec<u32>> = vec![
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![6, 7],
        vec![8, 9],
        vec![10, 11],
        vec![12, 13],
        vec![14, 15],
        vec![0, 2],
        vec![4, 6],
        vec![8, 10],
        vec![12, 14],
        vec![1, 3],
        vec![5, 7],
        vec![9, 11],
        vec![13, 15],
        vec![0, 8],
        vec![1, 9],
        vec![2, 10],
        vec![3, 11],
        vec![4, 12],
        vec![5, 13],
        vec![6, 14],
        vec![7, 15],
    ];
    // A verbatim duplicate of the first equation: after the original
    // eliminates, the copy's entry under the same pivot column has
    // already cancelled, so the probe hits the stale-listing `continue`.
    supports.push(supports[0].clone());
    // Weight-one rows on columns the singleton branch never pivots: their
    // listings stay live under other columns and probe pivots they do not
    // carry, exercising the singleton-path `continue`.
    supports.push(vec![0]);
    supports.push(vec![15]);
    assert_eq!(supports.len(), 27);

    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for support in &supports {
        hybrid.push_binary_row(support, &unit_payload(support, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&binary_system_of(n, sym, &supports, &x), &solved, Some(&x));
}

/// The cold list-merge path (`axpy_coeffs_slices` in `hybrid/sparse.rs`):
/// a field coefficient on a binary row widens it, and the widened row's
/// later elimination against a binary pivot merges explicit coefficients
/// through the out-of-place path. The widened update is observable: the
/// system still solves to the same rank, determinedness, and determined
/// values as a dense `Ple` over the same equations.
#[test]
fn widened_rows_merge_through_the_cold_list_path() {
    let (n, sym) = (16usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x60C3);
    let alpha = Gf8B::decode(&[0x1B]);
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    // A field row whose coefficients force the binary rows it touches to
    // widen: elimination against it leaves packed rows carrying explicit
    // coefficients, and their later binary-pivot merges run cold.
    let field_support: Vec<u32> = (0..8u32).collect();
    let field_coeffs: Vec<E> = (0..8)
        .map(|i| if i % 2 == 0 { alpha } else { E::ONE })
        .collect();
    let field_payload = payload(&field_support, &field_coeffs, &x, sym);
    hybrid.push_field_row(&field_support, &field_coeffs, &field_payload);
    // Binary rows overlapping the field row's support and reaching past
    // it: the merge both cancels shared columns and widens past them.
    let binary_supports: Vec<Vec<u32>> = vec![
        vec![0, 8],
        vec![1, 9],
        vec![2, 10],
        vec![3, 11],
        vec![4, 12],
        vec![5, 13],
        vec![6, 14],
        vec![7, 15],
        vec![8, 9],
        vec![10, 11],
        vec![12, 13],
        vec![14, 15],
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![6, 7],
    ];
    for support in &binary_supports {
        hybrid.push_binary_row(support, &unit_payload(support, &x, sym));
    }
    let solved = hybrid.solve();

    let mut rows: Vec<(Vec<u32>, Vec<E>)> = vec![(field_support, field_coeffs)];
    rows.extend(
        binary_supports
            .iter()
            .map(|support| (support.clone(), vec![E::ONE; support.len()])),
    );
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
}

/// The frozen-word half of the hot merge (`xor_frozen` in
/// `hybrid/sparse.rs`) only runs once columns have been inactivated into
/// bit words: a weight-two system with a scheduled inactivation merges
/// frozen words wholesale, including a word that fully cancels. The
/// observable claim is the same dense-`Ple` agreement, plus the reported
/// schedule showing the inactivation the frozen path requires.
#[test]
fn frozen_word_merge_matches_ple() {
    let (n, sym) = (24usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x60D4);
    // A stopping-set cluster over columns 0..6: every row touching the
    // cluster touches at least two of it, so weight-one peeling stalls
    // and the schedule inactivates before it can pivot. Rows pair cluster
    // columns with outside ones, so frozen words accumulate several
    // ordinals and later merges XOR multi-word spans.
    let cluster: Vec<u32> = (0..6u32).collect();
    let mut supports: Vec<Vec<u32>> = Vec::new();
    for i in 0..6u32 {
        for j in (i + 1)..6u32 {
            supports.push(vec![i, j]);
        }
    }
    for (i, &a) in cluster.iter().enumerate() {
        supports.push(vec![a, 6 + i as u32]);
        supports.push(vec![a, 12 + i as u32]);
    }
    for c in 6..n {
        let (mut a, mut b) = (c as u32, ((c + 3) % n) as u32);
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        if a != b {
            supports.push(vec![a, b]);
        }
    }
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for support in &supports {
        hybrid.push_binary_row(support, &unit_payload(support, &x, sym));
    }
    let (solved, stats) = hybrid.solve_reported(true).unwrap();
    assert!(
        stats.inactivations > 0,
        "the frozen path needs an inactivation"
    );
    assert_solution_matches_ple(
        &binary_system_of(n, sym, &supports, &x),
        &Ok(solved),
        Some(&x),
    );
}

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

#[test]
fn deferred_is_byte_identical_and_uses_fewer_row_ops() {
    let n = 32;
    let sym = 64;
    let x = random_solution::<Gf8B>(n, sym, 0xD3FE_0001);
    let mut sys = System::<Gf8B>::new(n, sym);
    for column in 0..n {
        let support = vec![column as u32];
        sys.push_consistent(support.clone(), vec![<Gf8B as Field>::Elem::ONE], &x);
    }
    // Redundant received equations: eager application updates each duplicate;
    // deferred replay omits them from the independent dense/pivot set.
    for repeat in 0..4 {
        for column in 0..n {
            let other = (column + repeat + 1) % n;
            let mut support = vec![column as u32, other as u32];
            support.sort_unstable();
            sys.push_consistent(support, vec![<Gf8B as Field>::Elem::ONE; 2], &x);
        }
    }

    let (eager_solution, eager) = sys.build_hybrid().solve_reported(false).unwrap();
    let (deferred_solution, deferred) = sys.build_hybrid().solve_reported(true).unwrap();
    assert_same_solution(&eager_solution, &deferred_solution, n);
    assert!(
        deferred.row_ops < eager.row_ops,
        "deferred={} eager={}",
        deferred.row_ops,
        eager.row_ops
    );
}

#[test]
fn field_band_widens_exactly_the_rows_it_touches() {
    let band = 8;
    let n = band * 2;
    let sym = 4;
    let x = random_solution::<Gf8B>(n, sym, 0xBADD_0001);
    let mut sys = System::<Gf8B>::new(n, sym);
    let alpha = <Gf8B as Field>::decode(&[2]);

    // All field pivots precede their binary targets. Each pivot touches one
    // distinct binary row, so exactly `band` transitions are necessary.
    for index in 0..band {
        sys.push_consistent(vec![(2 * index) as u32], vec![alpha], &x);
    }
    for index in 0..band {
        sys.push_consistent(
            vec![(2 * index) as u32, (2 * index + 1) as u32],
            vec![<Gf8B as Field>::Elem::ONE; 2],
            &x,
        );
    }

    let (solution, stats) = sys.build_hybrid().solve_reported(true).unwrap();
    assert!(solution.is_full_rank());
    assert_eq!(stats.widenings, band);
}

const RFC_DEGREE_THRESHOLDS: [u32; 31] = [
    0, 5_243, 529_531, 704_294, 791_675, 844_104, 879_057, 904_023, 922_747, 937_311, 948_962,
    958_494, 966_438, 973_160, 978_921, 983_914, 988_283, 992_138, 995_565, 998_631, 1_001_391,
    1_003_887, 1_006_157, 1_008_229, 1_010_129, 1_011_876, 1_013_490, 1_014_983, 1_016_370,
    1_017_662, 1_048_576,
];

fn rfc_degree(value: usize, columns: usize) -> usize {
    let degree = RFC_DEGREE_THRESHOLDS.partition_point(|&threshold| threshold <= value as u32);
    degree.min(columns.saturating_sub(2).max(1))
}

fn rfc_degree_system(columns: usize, seed: u64) -> System<Gf8B> {
    let mut state = seed;
    let overhead = (columns as f64).sqrt().ceil() as usize + 8;
    let mut system = System::<Gf8B>::new(columns, 1);
    for _ in 0..(columns + overhead) {
        let degree = rfc_degree(draw(&mut state, 1 << 20), columns);
        let support = support(columns, degree, &mut state);
        system.rows.push((
            support.clone(),
            vec![<Gf8B as Field>::Elem::ONE; support.len()],
            vec![0],
        ));
    }
    system
}

#[test]
fn rfc_degree_inactivation_scales_with_square_root() {
    let sizes = [10, 25, 50, 100, 250, 500, 1_000];
    let mut worst = 0.0f64;
    for &columns in &sizes {
        let mut maximum = 0;
        for seed in 0..8 {
            let (_, stats) = rfc_degree_system(columns, seed)
                .build_hybrid()
                .solve_reported(true)
                .unwrap();
            maximum = maximum.max(stats.inactivations);
        }
        let ratio = maximum as f64 / (columns as f64).sqrt();
        worst = worst.max(ratio);
        eprintln!("k={columns} max_g={maximum} g/sqrt(k)={ratio:.3}");
    }
    assert!(worst <= 3.0, "g/sqrt(k) grew to {worst:.3}");
}

/// A sorted, distinct random subset of `0..n`.
fn subset(n: usize, size: usize, st: &mut u64) -> Vec<u32> {
    let size = size.min(n);
    let mut pool: Vec<u32> = (0..n as u32).collect();
    for i in 0..size {
        let j = i + draw(st, n - i);
        pool.swap(i, j);
    }
    pool.truncate(size);
    pool.sort_unstable();
    pool
}

#[test]
fn initial_inactive_matches_ple() {
    // Random sparse systems with a random pre-inactivated subset, mixing
    // binary and field-valued rows, rectangular and rank-deficient shapes:
    // the answer must still match a dense Ple.
    for seed in 0..40u64 {
        let mut st = 0x6330_0000 ^ seed;
        let n = 8 + draw(&mut st, 40);
        let m = n / 2 + draw(&mut st, n);
        let sym = 1 + draw(&mut st, 3);
        let x = random_solution::<Gf8B>(n, sym, seed ^ 0x6330);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..m {
            let w = 1 + draw(&mut st, 5);
            let sup = support(n, w, &mut st);
            if draw(&mut st, 4) == 0 {
                let coeffs: Vec<<Gf8B as Field>::Elem> = sup
                    .iter()
                    .map(|_| <Gf8B as Field>::decode(&[(1 + draw(&mut st, 255)) as u8]))
                    .collect();
                sys.push_consistent(sup, coeffs, &x);
            } else {
                sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
            }
        }
        let initial = subset(n, draw(&mut st, n + 1), &mut st);
        assert_matches_ple_with(&sys, &initial);
    }
}

#[test]
fn all_initially_inactive_matches_ple() {
    // Every column starts inactive: the sparse phase peels nothing and the
    // whole solve is the dense block.
    for seed in 0..10u64 {
        let mut st = 0x0A11_0000 ^ seed;
        let n = 6 + draw(&mut st, 20);
        let sym = 2;
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
        for _ in 0..(n + 3) {
            let w = 1 + draw(&mut st, 4);
            let sup = support(n, w, &mut st);
            sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
        }
        let initial: Vec<u32> = (0..n as u32).collect();
        assert_matches_ple_with(&sys, &initial);
    }
}

#[test]
fn initial_inactive_inconsistent_systems_are_rejected() {
    for seed in 0..20u64 {
        let mut st = 0x1FC0_6330 ^ seed;
        let n = 10 + draw(&mut st, 20);
        let sym = 1;
        let x = random_solution::<Gf8B>(n, sym, seed);
        let mut sys = System::<Gf8B>::new(n, sym);
        for column in 0..n {
            sys.push_consistent(vec![column as u32], vec![<Gf8B as Field>::Elem::ONE], &x);
        }
        // The identity rows force a unique solution; a unit row on column 0
        // with a flipped payload makes the system inconsistent.
        sys.rows.push((
            vec![0],
            vec![<Gf8B as Field>::Elem::ONE],
            System::<Gf8B>::pack(&[x[0][0].add(<Gf8B as Field>::Elem::ONE)]),
        ));
        let initial = subset(n, draw(&mut st, n), &mut st);
        let mut hybrid = sys.build_hybrid_with(&initial);
        assert!(matches!(
            hybrid.solve(),
            Err(SolveError::Inconsistent { .. })
        ));
    }
}

#[test]
fn all_inactive_inconsistent_system_is_rejected() {
    // Every column starts inactive, so the whole solve is the dense block;
    // a corrupted duplicate row makes it inconsistent.
    let (n, sym) = (6usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x2E);
    let initial: Vec<u32> = (0..n as u32).collect();
    let mut hybrid = Hybrid::<Gf8B>::with_initial_inactive(n, sym, &initial);
    for c in 0..n {
        hybrid.push_binary_row(&[c as u32], &payload(&[c as u32], &[E::ONE], &x, sym));
    }
    let mut bad = payload(&[0], &[E::ONE], &x, sym);
    bad[0] ^= 0xFF;
    hybrid.push_binary_row(&[0], &bad);
    assert!(matches!(
        hybrid.solve(),
        Err(SolveError::Inconsistent { .. })
    ));
}

#[test]
fn initial_inactive_set_is_counted_once_and_stable_across_solves() {
    let n = 40;
    let sym = 8;
    let mut st = 0x0517_0001u64;
    let x = random_solution::<Gf8B>(n, sym, 0x0517_0002);
    let mut sys = System::<Gf8B>::new(n, sym);
    for column in 0..n {
        sys.push_consistent(vec![column as u32], vec![<Gf8B as Field>::Elem::ONE], &x);
    }
    for _ in 0..n {
        let sup = support(n, 3, &mut st);
        sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
    }
    let initial: Vec<u32> = (30..n as u32).collect();
    let mut hybrid = sys.build_hybrid_with(&initial);
    let (first_solution, first) = hybrid.solve_reported(true).unwrap();
    assert_eq!(first.initial_inactivations, initial.len());
    assert!(first.inactivations >= first.initial_inactivations);
    // Re-solving must reproduce the same schedule and answer, with every
    // initial column still inactive exactly once.
    let (second_solution, second) = hybrid.solve_reported(true).unwrap();
    assert_eq!(first, second);
    assert_same_solution(&first_solution, &second_solution, n);
}

/// An RFC-shaped system: RFC-degree LT rows over the leading `W` columns
/// plus a dense field-valued band over the trailing `band` columns (the
/// HDPC shape).
fn rfc_shaped_system(columns: usize, band: usize, seed: u64) -> System<Gf8B> {
    let mut state = seed;
    let lt_columns = columns - band;
    let overhead = (lt_columns as f64).sqrt().ceil() as usize + 8;
    let mut system = System::<Gf8B>::new(columns, 1);
    for _ in 0..(lt_columns + overhead) {
        let degree = rfc_degree(draw(&mut state, 1 << 20), lt_columns);
        let support = support(lt_columns, degree, &mut state);
        system.rows.push((
            support.clone(),
            vec![<Gf8B as Field>::Elem::ONE; support.len()],
            vec![0],
        ));
    }
    let alpha = <Gf8B as Field>::decode(&[2]);
    for index in 0..band {
        let support: Vec<u32> = ((lt_columns + index)..columns).map(|c| c as u32).collect();
        let mut coefficient = <Gf8B as Field>::Elem::ONE;
        let coeffs: Vec<_> = support
            .iter()
            .map(|_| {
                let c = coefficient;
                coefficient = coefficient.mul(alpha);
                c
            })
            .collect();
        system.rows.push((support, coeffs, vec![0]));
    }
    system
}

#[test]
fn rfc_shaped_pi_initialization_keeps_sqrt_dense_block() {
    // Pre-inactivating the trailing band on an RFC-shaped system keeps the
    // dense block at the sqrt scale the generic schedule achieves, and the
    // answer still matches a dense Ple.
    let sizes = [50usize, 100, 250];
    let mut worst = 0.0f64;
    for &columns in &sizes {
        let band = 8 + columns / 20;
        let initial: Vec<u32> = ((columns - band)..columns).map(|c| c as u32).collect();
        for seed in 0..4u64 {
            let sys = rfc_shaped_system(columns, band, 0x6330_0000 ^ seed);
            assert_matches_ple_with(&sys, &initial);
            let (_, stats) = sys
                .build_hybrid_with(&initial)
                .solve_reported(true)
                .unwrap();
            assert_eq!(stats.initial_inactivations, band);
            let ratio = stats.inactivations as f64 / (columns as f64).sqrt();
            worst = worst.max(ratio);
            eprintln!("columns={columns} band={band} g={}", stats.inactivations);
        }
    }
    assert!(worst <= 4.0, "g/sqrt(n) grew to {worst:.3}");
}

#[test]
#[should_panic(expected = "sorted and distinct")]
fn with_initial_inactive_rejects_unsorted() {
    _ = Hybrid::<Gf8B>::with_initial_inactive(4, 1, &[2, 2]);
}

#[test]
#[should_panic(expected = "out of range")]
fn with_initial_inactive_rejects_out_of_range() {
    _ = Hybrid::<Gf8B>::with_initial_inactive(4, 1, &[4]);
}

#[test]
fn sparse_deferred_row_matches_ple() {
    // Identity binary rows plus one sparse deferred field row on column 0:
    // most pivots never touch a deferred lane, and the binary pivot on
    // column 0 substitutes its own column out of the lane.
    let (n, sym) = (8usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0x3D);
    let alpha = Gf8B::decode(&[0x1B]);
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for c in 0..n {
        hybrid.push_binary_row(&[c as u32], &payload(&[c as u32], &[E::ONE], &x, sym));
    }
    hybrid.push_deferred_field_row(&[0], &[alpha], &payload(&[0], &[alpha], &x, sym));
    let solved = hybrid.solve().unwrap();
    assert!(solved.is_full_rank());
    for c in 0..n {
        let mut expected = vec![0u8; sym];
        for s in 0..sym {
            Gf8B::encode(&mut expected[s..=s], x[c][s]);
        }
        assert_eq!(solved.value(c), &expected[..], "column {c}");
    }
    // The eager push of the same row agrees.
    let mut eager = Hybrid::<Gf8B>::new(n, sym);
    for c in 0..n {
        eager.push_binary_row(&[c as u32], &payload(&[c as u32], &[E::ONE], &x, sym));
    }
    eager.push_field_row(&[0], &[alpha], &payload(&[0], &[alpha], &x, sym));
    let eager_solved = eager.solve().unwrap();
    assert_eq!(eager_solved.rank(), solved.rank());
    for c in 0..n {
        assert_eq!(eager_solved.value(c), solved.value(c));
    }
}

/// Deferred wide rows over an underdetermined system: release, emit, and
/// the kernel lift agree with the dense oracle.
#[test]
fn deferred_rows_over_deficient_system_match_ple() {
    let (n, sym) = (18usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0xD441);
    let alpha = Gf8B::decode(&[0x1B]);
    let mut rows: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for c in 0..10 {
        rows.push((vec![c as u32], vec![E::ONE]));
    }
    // Wide deferred rows confined to columns 0..12: the system stays
    // deficient and the release path emits over a singular block.
    let mut st = 0xD442u64;
    let mut deferred: Vec<(Vec<u32>, Vec<E>)> = Vec::new();
    for _ in 0..4 {
        let mut cols = Vec::new();
        while cols.len() < 8 {
            let c = draw(&mut st, 12) as u32;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        cols.sort_unstable();
        deferred.push((cols.clone(), vec![alpha; cols.len()]));
    }
    let mut all = rows.clone();
    all.extend(deferred.clone());
    let sys = system_of(n, sym, &all, &x);
    let (rank, ..) = ple_reference(&sys);
    assert!(rank < n, "test needs a deficient system, got rank {rank}");
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, coeffs) in &rows {
        hybrid.push_binary_row(support, &payload(support, coeffs, &x, sym));
    }
    for (support, coeffs) in &deferred {
        hybrid.push_deferred_field_row(support, coeffs, &payload(support, coeffs, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&sys, &solved, Some(&x));
}

/// Builds the solver with the last `band` rows deferred instead of eager.
fn build_hybrid_deferring(
    sys: &System<Gf8B>,
    band: usize,
    initial_inactive: &[u32],
) -> Hybrid<Gf8B> {
    let mut hybrid = Hybrid::<Gf8B>::with_initial_inactive(sys.n, sys.sym_len(), initial_inactive);
    let defer_from = sys.rows.len() - band;
    for (index, (support, coeffs, rhs)) in sys.rows.iter().enumerate() {
        if index >= defer_from {
            hybrid.push_deferred_field_row(support, coeffs, rhs);
        } else if coeffs.iter().all(|c| c.is_one()) {
            hybrid.push_binary_row(support, rhs);
        } else {
            hybrid.push_field_row(support, coeffs, rhs);
        }
    }
    hybrid
}

/// A system with a dense GF(2^8) band, the HDPC shape: LT-degree binary
/// rows over the leading columns plus `band` dense field rows over all
/// columns, with the trailing `band` columns pre-inactivated.
fn dense_band_system(
    n: usize,
    band: usize,
    seed: u64,
) -> (System<Gf8B>, Vec<u32>, Vec<Vec<<Gf8B as Field>::Elem>>) {
    let mut state = seed | 1;
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(2);
        for _s in 0..2 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            row.push(<Gf8B as Field>::decode(&[(state >> 33) as u8]));
        }
        x.push(row);
    }
    let mut sys = System::<Gf8B>::new(n, 2);
    for _ in 0..(n + 4) {
        let w = 1 + draw(&mut state, 4);
        let sup = support(n - band, w, &mut state);
        sys.push_consistent(sup.clone(), vec![<Gf8B as Field>::Elem::ONE; sup.len()], &x);
    }
    for _ in 0..band {
        let sup: Vec<u32> = (0..n as u32).collect();
        let coeffs: Vec<_> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                <Gf8B as Field>::decode(&[1 + (state >> 33) as u8 % 255])
            })
            .collect();
        sys.push_consistent(sup, coeffs, &x);
    }
    let initial: Vec<u32> = ((n - band)..n).map(|c| c as u32).collect();
    (sys, initial, x)
}

#[test]
fn deferred_dense_band_matches_eager_and_ple() {
    // Deferral changes the schedule, never the answer: the deferred solve
    // must match both the eager push and a dense Ple on rank, solution,
    // and determinedness across mixed, deficient, and full-rank systems.
    for seed in 0..30u64 {
        let mut st = 0xD3F3_0000 ^ seed;
        let n = 20 + draw(&mut st, 40);
        let band = 2 + draw(&mut st, 6);
        let (sys, initial, _x) = dense_band_system(n, band, seed);
        assert_matches_ple_with(&sys, &initial);
        let eager = sys.build_hybrid_with(&initial).solve().unwrap();
        let deferred = build_hybrid_deferring(&sys, band, &initial)
            .solve()
            .unwrap();
        assert_eq!(eager.rank(), deferred.rank(), "rank K seed {seed}");
        for column in 0..n {
            assert_eq!(
                eager.is_determined(column),
                deferred.is_determined(column),
                "determined {column} seed {seed}"
            );
            if eager.is_determined(column) {
                assert_eq!(
                    eager.value(column),
                    deferred.value(column),
                    "col {column} seed {seed}"
                );
            }
        }
    }
}

#[test]
fn deferred_dense_band_inconsistent_is_rejected() {
    let (mut sys, initial, x) = dense_band_system(24, 4, 0x1C_0001);
    // Identity rows pin the true solution; a deferred row with a
    // poisoned payload contradicts it, and the post-release
    // verification must catch the inconsistency.
    for column in 0..sys.n {
        sys.push_consistent(vec![column as u32], vec![<Gf8B as Field>::Elem::ONE], &x);
    }
    let mut poisoned = vec![0u8; sys.sym_len()];
    poisoned[0] ^= 1;
    sys.rows.push((
        (0..sys.n as u32).collect(),
        vec![<Gf8B as Field>::Elem::ONE; sys.n],
        poisoned,
    ));
    let mut hybrid = build_hybrid_deferring(&sys, 1, &initial);
    assert!(matches!(
        hybrid.solve(),
        Err(SolveError::Inconsistent { .. })
    ));
}

#[test]
fn deferred_rows_are_counted_and_releases_are_stable() {
    let (sys, initial, _x) = dense_band_system(40, 5, 0x5D_0001);
    let mut hybrid = build_hybrid_deferring(&sys, 5, &initial);
    let (first_solution, first) = hybrid.solve_reported(true).unwrap();
    assert_eq!(first.deferred_rows, 5);
    let (second_solution, second) = hybrid.solve_reported(true).unwrap();
    assert_eq!(first, second, "deferral release is deterministic");
    assert_same_solution(&first_solution, &second_solution, sys.n);
}

#[test]
fn more_than_lane_width_deferred_rows_still_solve() {
    // The batched release processes deferred rows in lane groups of the
    // accumulator width; a wider dense band must still match the dense
    // Ple, across group boundaries.
    for seed in 0..6u64 {
        let mut st = 0x0FF_0000 ^ seed;
        let band = 65 + draw(&mut st, 8);
        let n = 2 * band + 10 + draw(&mut st, 30);
        let (sys, initial, _x) = dense_band_system(n, band, seed);
        assert_matches_ple_with(&sys, &initial);
        let eager = sys.build_hybrid_with(&initial).solve().unwrap();
        let deferred = build_hybrid_deferring(&sys, band, &initial)
            .solve()
            .unwrap();
        assert_eq!(eager.rank(), deferred.rank());
        for column in 0..n {
            assert_eq!(eager.is_determined(column), deferred.is_determined(column));
            if eager.is_determined(column) {
                assert_eq!(eager.value(column), deferred.value(column));
            }
        }
    }
}

/// The allocation-free reported solve answers exactly like the allocating
/// one on the same system under the same reduction setting.
#[test]
fn solve_into_reported_matches_solve_reported() {
    let (n, sym) = (16usize, 3usize);
    let x = random_solution::<Gf8B>(n, sym, 0x9A);
    let build = |hybrid: &mut Hybrid<Gf8B>| {
        let mut st = 0x9A00u64;
        for _ in 0..(n + 6) {
            let w = 1 + draw(&mut st, 4);
            let mut cols = Vec::new();
            while cols.len() < w {
                let c = draw(&mut st, n) as u32;
                if !cols.contains(&c) {
                    cols.push(c);
                }
            }
            cols.sort_unstable();
            let coeffs = vec![E::ONE; cols.len()];
            hybrid.push_binary_row(&cols, &payload(&cols, &coeffs, &x, sym));
        }
    };

    let mut allocating = Hybrid::<Gf8B>::new(n, sym);
    build(&mut allocating);
    let (expected_solution, expected_stats) = allocating.solve_reported(true).unwrap();

    let mut in_place = Hybrid::<Gf8B>::new(n, sym);
    build(&mut in_place);
    let mut values = Matrix::<Gf8B>::zeros(n, sym).unwrap();
    let mut determined = vec![false; n];
    let stats = in_place
        .solve_into_reported(true, &mut values, &mut determined)
        .unwrap();

    assert_eq!(stats.rank, expected_stats.rank);
    assert_eq!(stats.rank, expected_solution.rank());
    for c in 0..n {
        assert_eq!(
            determined[c],
            expected_solution.is_determined(c),
            "column {c}"
        );
        assert_eq!(values.row(c), expected_solution.value(c), "column {c}");
    }
}

/// A payload-only re-solve must be the solve it replaces: same rank, same
/// determinedness, same bytes — including through a deferred dense band.
#[test]
fn replayed_solve_matches_a_fresh_solve() {
    let (n, band) = (120usize, 12usize);
    let (sys, initial, _x) = dense_band_system(n, band, 0x4E9A_0011);
    let y = random_solution::<Gf8B>(n, sys.sym, 0x4E9A_0022);
    let mut updated = System::<Gf8B>::new(n, sys.sym);
    for (support, coeffs, _) in &sys.rows {
        updated.push_consistent(support.clone(), coeffs.clone(), &y);
    }

    let mut solver = build_hybrid_deferring(&sys, band, &initial);
    let mut values = Matrix::<Gf8B>::zeros(n, sys.sym).unwrap();
    let mut determined = vec![false; n];
    solver.solve_into(&mut values, &mut determined).unwrap();
    for (row, (_, _, rhs)) in updated.rows.iter().enumerate() {
        solver.replace_rhs(row, rhs);
    }
    let replayed = solver.resolve_into(&mut values, &mut determined).unwrap();

    let expected = build_hybrid_deferring(&updated, band, &initial)
        .solve()
        .unwrap();
    assert_eq!(replayed, expected.rank());
    for column in 0..n {
        assert_eq!(determined[column], expected.is_determined(column));
        assert_eq!(values.row(column), expected.value(column));
    }
}

/// The replay path verifies the rows outside the independent set against
/// the new payloads, so it rejects exactly what a fresh solve rejects.
#[test]
fn replayed_solve_reports_the_same_inconsistency() {
    let (n, band) = (60usize, 6usize);
    let (mut sys, initial, _x) = dense_band_system(n, band, 0x4E9A_0033);
    // A duplicate of the first equation: redundant while its payload
    // agrees, contradictory the moment it does not.
    let (support, coeffs, _) = sys.rows[0].clone();
    sys.rows.push((support, coeffs, sys.rows[0].2.clone()));
    let duplicate = sys.rows.len() - 1;

    let mut solver = build_hybrid_deferring(&sys, band, &initial);
    let mut values = Matrix::<Gf8B>::zeros(n, sys.sym).unwrap();
    let mut determined = vec![false; n];
    solver.solve_into(&mut values, &mut determined).unwrap();

    let mut corrupt = sys.rows[duplicate].2.clone();
    corrupt[0] ^= 0xFF;
    solver.replace_rhs(duplicate, &corrupt);
    let replayed = solver.resolve_into(&mut values, &mut determined);

    sys.rows[duplicate].2 = corrupt;
    let fresh = build_hybrid_deferring(&sys, band, &initial).solve();
    match (replayed, fresh) {
        (Err(SolveError::Inconsistent { row: left }), Err(SolveError::Inconsistent { row })) => {
            assert_eq!(left, row);
        }
        other => panic!("expected matching inconsistency, got {other:?}"),
    }
}

/// The single-threaded Rayon boundary behind `internals`: with a
/// two-megabyte row the dispatcher would parallelize, but a one-thread
/// pool takes the serial fallback — and the arithmetic is identical.
#[cfg(feature = "parallel")]
#[test]
fn single_thread_pool_takes_the_serial_row_update() {
    use crate::common::noise;
    use gfm::internals::row_ops;
    let len = 2 * 1024 * 1024 + 64;
    let factor = Gf8B::decode(&[0x53]);
    let src = noise(len, 0xBA11);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(rayon::current_num_threads(), 1);
        let mut dst = noise(len, 0xBA12);
        let expected_first = Gf8B::decode(&dst[0..1]).add(factor.mul(Gf8B::decode(&src[0..1])));
        row_ops::mul_add::<Gf8B>(&mut dst, factor, &src);
        assert_eq!(Gf8B::decode(&dst[0..1]), expected_first);
        // The whole row matches the scalar reference computed inline.
        let mut reference = noise(len, 0xBA12);
        for i in 0..len {
            let d = Gf8B::decode(&reference[i..=i]);
            let s = Gf8B::decode(&src[i..=i]);
            Gf8B::encode(&mut reference[i..=i], d.add(factor.mul(s)));
        }
        assert_eq!(dst, reference);
    });
}

/// A narrow system with wide binary rows: column listings stay non-singleton,
/// so binary pivots merge through the packed XOR path instead of the
/// singleton peel.
#[test]
fn narrow_binary_system_merges_through_the_packed_xor_path() {
    let (n, sym) = (5usize, 2usize);
    let x = random_solution::<Gf8B>(n, sym, 0xE501);
    let rows: Vec<(Vec<u32>, Vec<E>)> = [
        vec![0, 1],
        vec![0, 1, 2],
        vec![1, 3],
        vec![2, 4],
        vec![3, 4],
        vec![0, 2, 4],
    ]
    .iter()
    .map(|support| (support.clone(), vec![E::ONE; support.len()]))
    .collect();
    let mut hybrid = Hybrid::<Gf8B>::new(n, sym);
    for (support, _) in &rows {
        hybrid.push_binary_row(support, &unit_payload(support, &x, sym));
    }
    let solved = hybrid.solve();
    assert_solution_matches_ple(&system_of(n, sym, &rows, &x), &solved, Some(&x));
}
