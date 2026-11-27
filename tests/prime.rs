//! The public generic operations over the prime fields, against independent
//! oracles. Every elimination-sensitive path — the dense PLE decomposition and
//! its derived queries, triangular solves, matrix multiply, the compact
//! solver, the hybrid inactivation solver, and the streaming echelon — runs
//! over `Mersenne31`, `Goldilocks`, and `QuadMersenne31`, where addition and
//! subtraction differ and elimination signs are observable. The binary-field
//! suites cover the characteristic-two behavior of the same paths.

mod common;
mod oracles;

use common::draw;
use fgf::extension::QuadMersenne31;
use fgf::prime::{Goldilocks, Mersenne31, PrimeIdentity};
use fgf::{Elem, FieldElem};
use fgf::{Field, FieldBuffer};
use gfm::dense::layout::ALIGN;
use gfm::{
    Echelon, Hybrid, Innovation, Matrix, Ple, PleScratch, SmallMatrix, SolveError, SolveScratch,
    mul_add, mul_into, solve_lower_unit_assign, solve_upper_assign,
};
use oracles::{
    Naive, matrix_of, naive_det, naive_identity, naive_mul, naive_of, oracle_ple, oracle_rref,
    pack, reassemble,
};

/// A prime field with a deterministic canonical element stream, so oracle
/// comparisons see one spelling per value; aliases have their own suite in
/// `tests/noncanonical.rs`.
trait PrimeField: FieldBuffer {
    /// One canonical element, zero included, drawn from `state`.
    fn elem(state: &mut u64) -> Elem<Self>;
}

impl PrimeField for Mersenne31 {
    fn elem(state: &mut u64) -> Elem<Self> {
        Elem::<Self>::from_raw(draw(
            state,
            <<Mersenne31 as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC as usize,
        ) as u32)
    }
}

impl PrimeField for Goldilocks {
    fn elem(state: &mut u64) -> Elem<Self> {
        // Two draws under 2^30 combine into a lane below the modulus.
        let lo = draw(state, 1 << 30) as u64;
        let hi = draw(state, 1 << 30) as u64;
        Elem::<Self>::from_raw((hi << 30) | lo)
    }
}

impl PrimeField for QuadMersenne31 {
    fn elem(state: &mut u64) -> Elem<Self> {
        let re = Elem::<Mersenne31>::from_raw(draw(state, u32::MAX as usize) as u32).to_raw();
        let im = Elem::<Mersenne31>::from_raw(draw(state, u32::MAX as usize) as u32).to_raw();
        Elem::<Self>::from_raw([re, im])
    }
}

/// A canonical noise matrix over `F`.
fn naive_field<F: PrimeField>(rows: usize, cols: usize, seed: u64) -> Naive<F> {
    let mut state = seed | 1;
    (0..rows)
        .map(|_| (0..cols).map(|_| F::elem(&mut state)).collect())
        .collect()
}

/// The `m × r` unit lower triangle with canonical noise below the diagonal.
fn unit_lower<F: PrimeField>(rows: usize, cols: usize, state: &mut u64) -> Naive<F> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|t| {
                    if t < i {
                        F::elem(state)
                    } else if t == i {
                        Elem::<F>::ONE
                    } else {
                        Elem::<F>::ZERO
                    }
                })
                .collect()
        })
        .collect()
}

/// The `r × n` unit upper triangle with canonical noise above the diagonal.
fn unit_upper<F: PrimeField>(rows: usize, cols: usize, state: &mut u64) -> Naive<F> {
    (0..rows)
        .map(|t| {
            (0..cols)
                .map(|c| {
                    if c > t {
                        F::elem(state)
                    } else if c == t {
                        Elem::<F>::ONE
                    } else {
                        Elem::<F>::ZERO
                    }
                })
                .collect()
        })
        .collect()
}

/// A nonsingular `n × n` matrix: the product of two unit triangles.
fn naive_invertible<F: PrimeField>(n: usize, seed: u64) -> Naive<F> {
    let mut state = seed | 1;
    let l = unit_lower::<F>(n, n, &mut state);
    let u = unit_upper::<F>(n, n, &mut state);
    naive_mul::<F>(&l, &u)
}

/// An exact-rank `rows × cols` matrix: `L·U` with `L` unit lower `m × r` and
/// `U` unit upper `r × n`. The leading `r × r` minor of the product has
/// determinant one, so the rank is exactly `r`.
fn naive_rank_of<F: PrimeField>(rows: usize, cols: usize, rank: usize, seed: u64) -> Naive<F> {
    assert!(rank <= rows.min(cols));
    if rank == 0 {
        return (0..rows).map(|_| vec![Elem::<F>::ZERO; cols]).collect();
    }
    let mut state = seed | 1;
    let l = unit_lower::<F>(rows, rank, &mut state);
    let u = unit_upper::<F>(rank, cols, &mut state);
    naive_mul::<F>(&l, &u)
}

/// Gauss–Jordan solve of a nonsingular square system, one pivot at a time.
// The augmented row spans coefficients plus right-hand sides; the split
// index `n` is the algorithm.
#[allow(clippy::needless_range_loop)]
fn naive_solve<F: FieldBuffer>(a: &Naive<F>, b: &Naive<F>) -> Naive<F> {
    let n = a.len();
    let s = b.first().map_or(0, Vec::len);
    let mut aug: Naive<F> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend(b[i].iter().copied());
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .find(|&r| !aug[r][col].is_zero())
            .expect("nonsingular system");
        aug.swap(piv, col);
        let inv = aug[col][col].inv();
        for cell in aug[col].iter_mut() {
            *cell = cell.mul(inv);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug[r][col];
            if f.is_zero() {
                continue;
            }
            for c in 0..(n + s) {
                aug[r][c] = aug[r][c].sub(f.mul(aug[col][c]));
            }
        }
    }
    aug.into_iter().map(|row| row[n..].to_vec()).collect()
}

/// Evaluates one sparse equation against per-column solution rows.
fn eval_eq<F: FieldBuffer>(
    support: &[u32],
    coeffs: &[Elem<F>],
    sol: &[Vec<Elem<F>>],
) -> Vec<Elem<F>> {
    let s = sol.first().map_or(0, Vec::len);
    let mut acc = vec![Elem::<F>::ZERO; s];
    for (&col, &c) in support.iter().zip(coeffs) {
        for (k, slot) in acc.iter_mut().enumerate() {
            *slot = slot.add(c.mul(sol[col as usize][k]));
        }
    }
    acc
}

/// The dense coefficient matrix of pushed sparse equations.
fn dense_of_system<F: FieldBuffer>(rows: &[(Vec<u32>, Vec<Elem<F>>)], n: usize) -> Naive<F> {
    let mut out: Naive<F> = (0..rows.len()).map(|_| vec![Elem::<F>::ZERO; n]).collect();
    for (i, (support, coeffs)) in rows.iter().enumerate() {
        for (&col, &c) in support.iter().zip(coeffs) {
            out[i][col as usize] = c;
        }
    }
    out
}

/// Byte width of `sym_cols` field elements.
fn sym_bytes<F: FieldBuffer>(sym_cols: usize) -> usize {
    sym_cols * F::BYTES
}

/// The dense reference verdict for a hybrid system: rank, consistency, the
/// packed solution, and per-column determinedness from the kernel support.
fn hybrid_reference<F: PrimeField>(
    rows: &[(Vec<u32>, Vec<Elem<F>>)],
    rhs: &[Vec<u8>],
    n: usize,
    sym_cols: usize,
) -> (usize, bool, Option<Matrix<F>>, Vec<bool>) {
    let dense = dense_of_system::<F>(rows, n);
    let ple = Ple::decompose(to_matrix::<F>(&dense, n), &mut PleScratch::new());
    let rank = ple.rank();
    let mut kernel = Matrix::<F>::zeros(n, n - rank).unwrap();
    ple.kernel_into(&mut kernel);
    let determined = (0..n)
        .map(|c| (0..n - rank).all(|j| kernel.get(c, j).is_zero()))
        .collect();
    let b: Naive<F> = rhs
        .iter()
        .map(|bytes| {
            (0..sym_cols)
                .map(|k| F::decode(&bytes[k * F::BYTES..(k + 1) * F::BYTES]))
                .collect()
        })
        .collect();
    let mut out = Matrix::<F>::zeros(n, sym_cols).unwrap();
    match ple.solve_into(
        &to_matrix::<F>(&b, sym_cols),
        &mut out,
        &mut SolveScratch::new(),
    ) {
        Ok(()) => (rank, true, Some(out), determined),
        Err(SolveError::Inconsistent { .. }) => (rank, false, None, determined),
        Err(e) => panic!("unexpected reference solve error: {e:?}"),
    }
}

/// Packs one naive row into its little-endian bytes.
fn pack_row<F: FieldBuffer>(row: &[Elem<F>]) -> Vec<u8> {
    pack::<F>(&vec![row.to_vec()]).pop().unwrap()
}

/// Asserts the packed solution satisfies every equation exactly.
fn assert_system_satisfied<F: PrimeField>(
    rows: &[(Vec<u32>, Vec<Elem<F>>)],
    rhs: &[Vec<u8>],
    sol: &[Vec<Elem<F>>],
) {
    for (i, (support, coeffs)) in rows.iter().enumerate() {
        let expected = pack_row::<F>(&eval_eq::<F>(support, coeffs, sol));
        assert_eq!(expected, rhs[i], "equation {i} unsatisfied");
    }
}

/// Packs naive rows into a crate matrix of the given column count. `cols` is
/// explicit because an empty row list carries no column count of its own.
fn to_matrix<F: FieldBuffer>(a: &Naive<F>, cols: usize) -> Matrix<F> {
    matrix_of::<F>(&pack::<F>(a), cols)
}

/// A sorted random support of the drawn size inside `0..n`.
fn random_support(n: usize, state: &mut u64) -> Vec<u32> {
    let want = (*state % 5 + 2) as usize;
    let mut s: Vec<u32> = (0..want).map(|_| draw(state, n) as u32).collect();
    s.sort_unstable();
    s.dedup();
    if s.is_empty() {
        s.push(0);
    }
    s
}

/// Field coefficients parallel to `support`, never zero: a zero would not be
/// part of a support.
fn random_coeffs<F: PrimeField>(len: usize, state: &mut u64) -> Vec<Elem<F>> {
    (0..len)
        .map(|_| {
            let c = F::elem(state);
            if c.is_zero() { Elem::<F>::ONE } else { c }
        })
        .collect()
}

// ── Dense decomposition and derived queries ─────────────────────────────────

/// The full dense sweep for one prime field: rank, factor storage, RREF,
/// kernel, and determinant against the independent oracles.
fn check_ple<F: PrimeField>() {
    let mut dims = vec![
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 7),
        (7, 4),
        (8, 8),
        (9, 6),
    ];
    // Column counts whose byte width straddles the alignment boundary.
    let base = ALIGN / F::BYTES;
    for cols in [base - 1, base, base + 1] {
        dims.push((1, cols));
        dims.push((3, cols));
    }
    for &(rows, cols) in &dims {
        for rank in 0..=rows.min(cols) {
            let seed = 0x1000 ^ ((rows as u64) << 16) ^ ((cols as u64) << 8) ^ (rank as u64);
            let a = naive_rank_of::<F>(rows, cols, rank, seed);
            let o = oracle_ple::<F>(&a);
            let ple = Ple::decompose(to_matrix::<F>(&a, cols), &mut PleScratch::new());
            assert_eq!(ple.rank(), o.rank, "rank at ({rows}, {cols}, r{rank})");
            assert_eq!(reassemble(&o), a, "oracle certificate");

            let mut rref = Matrix::<F>::zeros(rows, cols).unwrap();
            ple.rref_into(&mut rref);
            assert_eq!(
                naive_of::<F>(&rref),
                oracle_rref::<F>(&a),
                "rref at ({rows}, {cols})"
            );

            if rows > 0 && cols > rank {
                let mut kernel = Matrix::<F>::zeros(cols, cols - rank).unwrap();
                ple.kernel_into(&mut kernel);
                let product = naive_mul::<F>(&a, &naive_of::<F>(&kernel));
                assert!(
                    product
                        .iter()
                        .all(|row| row.iter().all(|cell| cell.is_zero())),
                    "A·kernel must vanish at ({rows}, {cols}, r{rank})"
                );
            }

            if rows == cols {
                assert_eq!(ple.det(), naive_det::<F>(&a), "det at ({rows}, {cols})");
            }

            let lu = naive_of::<F>(&ple.into_matrix());
            assert_eq!(lu, o.lu, "factor storage at ({rows}, {cols}, r{rank})");
        }
    }

    // The reversed identity is an odd permutation: its determinant is minus
    // one, and the decomposition reproduces the signed value.
    let a: Naive<F> = vec![
        vec![Elem::<F>::ZERO, Elem::<F>::ONE],
        vec![Elem::<F>::ONE, Elem::<F>::ZERO],
    ];
    let ple = Ple::decompose(to_matrix::<F>(&a, 2), &mut PleScratch::new());
    assert_eq!(naive_det::<F>(&a), Elem::<F>::ONE.neg());
    assert_eq!(ple.det(), Elem::<F>::ONE.neg());
}

/// Solves over one prime field: unique solutions against the textbook
/// oracle, rank-deficient consistency, and the inconsistent verdict.
fn check_solve<F: PrimeField>() {
    for n in 1..=6 {
        let s = 1 + (n % 3);
        let a = naive_invertible::<F>(n, 0x5101 ^ n as u64);
        let x = naive_field::<F>(n, s, 0x5E01 ^ n as u64);
        let b = naive_mul::<F>(&a, &x);
        let ple = Ple::decompose(to_matrix::<F>(&a, n), &mut PleScratch::new());
        let mut out = Matrix::<F>::zeros(n, s).unwrap();
        ple.solve_into(&to_matrix::<F>(&b, s), &mut out, &mut SolveScratch::new())
            .unwrap();
        assert_eq!(naive_of::<F>(&out), naive_solve::<F>(&a, &b), "n={n}");
        assert_eq!(naive_of::<F>(&out), x, "unique solution at n={n}");
    }

    // Rank-deficient consistent systems: a solution is produced and every
    // equation holds.
    let a = naive_rank_of::<F>(5, 7, 3, 0x51D1);
    let x = naive_field::<F>(7, 2, 0x51D2);
    let b = naive_mul::<F>(&a, &x);
    let ple = Ple::decompose(to_matrix::<F>(&a, 7), &mut PleScratch::new());
    let mut out = Matrix::<F>::zeros(7, 2).unwrap();
    ple.solve_into(&to_matrix::<F>(&b, 2), &mut out, &mut SolveScratch::new())
        .unwrap();
    let solved = naive_of::<F>(&out);
    assert_eq!(naive_mul::<F>(&a, &solved), b);

    // Zero-row edge: a 0 × 0 system solves trivially.
    let ple = Ple::decompose(Matrix::<F>::zeros(0, 0).unwrap(), &mut PleScratch::new());
    ple.solve_into(
        &Matrix::<F>::zeros(0, 0).unwrap(),
        &mut Matrix::<F>::zeros(0, 0).unwrap(),
        &mut SolveScratch::new(),
    )
    .unwrap();

    // An inconsistent system: the eliminated extra row is zero on the left
    // and nonzero on the right, and the oracle's ranks agree.
    let a = naive_invertible::<F>(4, 0x51AA);
    let b = naive_field::<F>(4, 2, 0x51BB);
    let mut tall_a = a.clone();
    let mut tall_b = b.clone();
    tall_a.push(a[0].clone());
    tall_b.push(b[0].iter().map(|e| e.add(Elem::<F>::ONE)).collect());
    let aug: Naive<F> = tall_a
        .iter()
        .zip(&tall_b)
        .map(|(l, r)| {
            let mut row = l.clone();
            row.extend(r.iter().copied());
            row
        })
        .collect();
    assert_eq!(
        oracle_ple::<F>(&tall_a).rank,
        oracle_ple::<F>(&aug).rank - 1,
        "the oracle must agree the augmented system is inconsistent"
    );
    let ple = Ple::decompose(to_matrix::<F>(&tall_a, 4), &mut PleScratch::new());
    let err = ple
        .solve_into(
            &to_matrix::<F>(&tall_b, 2),
            &mut Matrix::<F>::zeros(4, 2).unwrap(),
            &mut SolveScratch::new(),
        )
        .unwrap_err();
    assert!(matches!(err, SolveError::Inconsistent { .. }));
}

/// Inverses over one prime field: both products are the identity.
fn check_inverse<F: PrimeField>() {
    for n in 1..=6 {
        let a = naive_invertible::<F>(n, 0x1A50 + n as u64);
        let ple = Ple::decompose(to_matrix::<F>(&a, n), &mut PleScratch::new());
        let mut inv = Matrix::<F>::zeros(n, n).unwrap();
        ple.inverse_into(&mut inv).unwrap();
        let inv_naive = naive_of::<F>(&inv);
        assert_eq!(
            naive_mul::<F>(&a, &inv_naive),
            naive_identity::<F>(n),
            "A·A⁻¹ at n={n}"
        );
        assert_eq!(
            naive_mul::<F>(&inv_naive, &a),
            naive_identity::<F>(n),
            "A⁻¹·A at n={n}"
        );
    }

    // Rank deficiency is the Singular error.
    let mut a = naive_invertible::<F>(3, 0x1A00);
    a[2] = vec![Elem::<F>::ZERO; 3];
    let ple = Ple::decompose(to_matrix::<F>(&a, 3), &mut PleScratch::new());
    let err = ple
        .inverse_into(&mut Matrix::<F>::zeros(3, 3).unwrap())
        .unwrap_err();
    assert!(matches!(err, SolveError::Singular { .. }));
}

/// Determinants over one prime field against cofactor expansion.
fn check_det<F: PrimeField>() {
    for n in 0..=5 {
        let a = naive_field::<F>(n, n, 0xDE17 ^ n as u64);
        let ple = Ple::decompose(to_matrix::<F>(&a, n), &mut PleScratch::new());
        assert_eq!(ple.det(), naive_det::<F>(&a), "det at n={n}");
    }

    // A singular matrix has determinant zero on both sides.
    let mut a = naive_field::<F>(3, 3, 0xDE00);
    a[2] = a[0].clone();
    let ple = Ple::decompose(to_matrix::<F>(&a, 3), &mut PleScratch::new());
    assert!(ple.det().is_zero());
    assert!(naive_det::<F>(&a).is_zero());

    // Odd permutation: the reversed identity carries determinant minus one.
    let a: Naive<F> = vec![
        vec![Elem::<F>::ZERO, Elem::<F>::ONE],
        vec![Elem::<F>::ONE, Elem::<F>::ZERO],
    ];
    let ple = Ple::decompose(to_matrix::<F>(&a, 2), &mut PleScratch::new());
    assert_eq!(ple.det(), Elem::<F>::ONE.neg());
    assert_eq!(naive_det::<F>(&a), Elem::<F>::ONE.neg());
}

// ── Row kernels ──────────────────────────────────────────────────────────────

/// Matrix multiply over one prime field against the naive oracle, empty
/// shapes included.
fn check_gemm<F: PrimeField>() {
    for (m, k, n) in [
        (0usize, 3usize, 2usize),
        (3, 0, 2),
        (2, 3, 0),
        (2, 3, 4),
        (7, 5, 6),
    ] {
        let a = naive_field::<F>(m, k, 0x6E21 ^ ((m as u64) << 8) ^ (k as u64));
        let b = naive_field::<F>(k, n, 0x6E22 ^ ((k as u64) << 8) ^ (n as u64));
        // The oracle cannot read a dimension from an empty operand: a
        // zero-row product is the empty matrix, a zero-length product is the
        // zero matrix.
        let product: Naive<F> = if m == 0 {
            Vec::new()
        } else if k == 0 {
            (0..m).map(|_| vec![Elem::<F>::ZERO; n]).collect()
        } else {
            naive_mul::<F>(&a, &b)
        };

        let mut out = Matrix::<F>::zeros(m, n).unwrap();
        mul_into(&mut out, &to_matrix::<F>(&a, k), &to_matrix::<F>(&b, n));
        assert_eq!(
            naive_of::<F>(&out),
            product,
            "overwrite shape ({m},{k},{n})"
        );

        // Accumulation adds onto the previous contents.
        let seed = naive_field::<F>(m, n, 0x6E23);
        let mut acc = to_matrix::<F>(&seed, n);
        mul_add(&mut acc, &to_matrix::<F>(&a, k), &to_matrix::<F>(&b, n));
        let sum: Naive<F> = seed
            .iter()
            .zip(&product)
            .map(|(l, r)| l.iter().zip(r).map(|(x, y)| x.add(*y)).collect())
            .collect();
        assert_eq!(naive_of::<F>(&acc), sum, "accumulate shape ({m},{k},{n})");
    }
}

/// Triangular solves over one prime field, verified by naive products.
fn check_trsm<F: PrimeField>() {
    let n = 5;
    let mut state = 0x7E51 | 1;
    let l = unit_lower::<F>(n, n, &mut state);
    let mut u = unit_upper::<F>(n, n, &mut state);
    // The upper triangle needs a nonzero diagonal.
    for (i, row) in u.iter_mut().enumerate() {
        if row[i].is_zero() {
            row[i] = Elem::<F>::ONE;
        }
    }
    let b = naive_field::<F>(n, 3, 0x7E52);

    let mut x = to_matrix::<F>(&b, 3);
    solve_lower_unit_assign(&mut x, &to_matrix::<F>(&l, n));
    assert_eq!(naive_mul::<F>(&l, &naive_of::<F>(&x)), b, "L·X = B");

    let mut y = to_matrix::<F>(&b, 3);
    solve_upper_assign(&mut y, &to_matrix::<F>(&u, n));
    assert_eq!(naive_mul::<F>(&u, &naive_of::<F>(&y)), b, "U·X = B");
}

// ── Compact solver ───────────────────────────────────────────────────────────

/// The compact solver over one prime field at fixed order `K`.
fn check_small_k<F: PrimeField, const K: usize>() {
    let a = naive_invertible::<F>(K, 0x5A11 ^ K as u64);
    let b = naive_field::<F>(K, 3, 0x50F1 ^ K as u64);
    let small = SmallMatrix::<F, K>::from_matrix(&to_matrix::<F>(&a, K));
    assert_eq!(small.rank(), K);

    let mut out = Matrix::<F>::zeros(K, 3).unwrap();
    small.solve_into(&to_matrix::<F>(&b, 3), &mut out).unwrap();
    assert_eq!(naive_of::<F>(&out), naive_solve::<F>(&a, &b), "K={K}");

    // A rank-deficient block answers Singular.
    let mut sing = a.clone();
    sing[K - 1] = vec![Elem::<F>::ZERO; K];
    let small = SmallMatrix::<F, K>::from_matrix(&to_matrix::<F>(&sing, K));
    let err = small
        .solve_into(&to_matrix::<F>(&b, 3), &mut out)
        .unwrap_err();
    assert!(matches!(err, SolveError::Singular { .. }), "K={K}");
}

fn check_small<F: PrimeField>() {
    check_small_k::<F, 1>();
    check_small_k::<F, 2>();
    check_small_k::<F, 3>();
    check_small_k::<F, 4>();
    check_small_k::<F, 5>();
    check_small_k::<F, 6>();
}

// ── Hybrid sparse/dense solver ───────────────────────────────────────────────

/// One mixed hybrid system over a known solution: binary rows (a duplicated
/// support included), field rows, and deferred rows.
fn hybrid_consistent_case<F: PrimeField>(n: usize, sym_cols: usize, initial: &[u32]) {
    let sym_len = sym_bytes::<F>(sym_cols);
    let mut state = (0x6E10 ^ n as u64) | 1;
    let x: Vec<Vec<Elem<F>>> = (0..n)
        .map(|_| (0..sym_cols).map(|_| F::elem(&mut state)).collect())
        .collect();

    let mut rows: Vec<(Vec<u32>, Vec<Elem<F>>)> = Vec::new();
    let mut rhs: Vec<Vec<u8>> = Vec::new();

    // Binary rows, two of them on the same support: eliminating one against
    // the other must cancel them exactly, which packed XOR-assumption code
    // would get right by accident but widened arithmetic must reproduce.
    let shared: Vec<u32> = (0..4).map(|c| (c * 3) as u32).collect();
    for support in [
        shared.clone(),
        shared,
        (1..6).map(|c| (c * 2) as u32).collect(),
    ] {
        let coeffs = vec![Elem::<F>::ONE; support.len()];
        rhs.push(pack_row::<F>(&eval_eq::<F>(&support, &coeffs, &x)));
        rows.push((support, coeffs));
    }
    // Field rows with canonical noise coefficients.
    for _ in 0..6 {
        let support = random_support(n, &mut state);
        let coeffs = random_coeffs::<F>(support.len(), &mut state);
        rhs.push(pack_row::<F>(&eval_eq::<F>(&support, &coeffs, &x)));
        rows.push((support, coeffs));
    }

    let mut hybrid = Hybrid::<F>::with_initial_inactive(n, sym_len, initial);
    for (i, (support, coeffs)) in rows.iter().enumerate() {
        // `push_binary_row` asserts all-ones coefficients; the random rows
        // carry explicit coefficients and go through the field push.
        let all_ones = coeffs.iter().all(|c| c.is_one());
        if !all_ones && i % 3 == 2 {
            hybrid.push_deferred_field_row(support, coeffs, &rhs[i]);
        } else if all_ones && i % 2 == 0 {
            hybrid.push_binary_row(support, &rhs[i]);
        } else {
            hybrid.push_field_row(support, coeffs, &rhs[i]);
        }
    }
    let mut values = Matrix::<F>::zeros(n, sym_cols).unwrap();
    let mut determined = vec![false; n];
    let rank = hybrid.solve_into(&mut values, &mut determined).unwrap();

    // Rank, consistency, determinedness, and determined values match the
    // dense reference; the answer satisfies every original equation.
    let (rank_ref, consistent, sol_ref, determined_ref) =
        hybrid_reference::<F>(&rows, &rhs, n, sym_cols);
    assert!(consistent, "the case must be consistent");
    assert_eq!(rank, rank_ref, "rank against the dense reference");
    for (col, &expected) in determined_ref.iter().enumerate() {
        assert_eq!(determined[col], expected, "determinedness at col {col}");
        if expected {
            assert_eq!(values.row(col), sol_ref.as_ref().unwrap().row(col));
        }
    }
    let sol = naive_of::<F>(&values);
    assert_system_satisfied::<F>(&rows, &rhs, &sol);
}

/// Rank-deficient and inconsistent hybrid verdicts over one prime field.
fn hybrid_deficient_and_inconsistent_case<F: PrimeField>() {
    let n = 8usize;
    let sym_cols = 2usize;
    let sym_len = sym_bytes::<F>(sym_cols);
    let mut state = 0x6E77 | 1;
    let x: Vec<Vec<Elem<F>>> = (0..n)
        .map(|_| (0..sym_cols).map(|_| F::elem(&mut state)).collect())
        .collect();

    // Equations only over the first six columns: the last two stay free.
    // Even rows are all-ones equations, odd rows carry explicit coefficients.
    let mut rows: Vec<(Vec<u32>, Vec<Elem<F>>)> = Vec::new();
    let mut rhs: Vec<Vec<u8>> = Vec::new();
    for i in 0..6usize {
        let support: Vec<u32> = (0..6)
            .filter(|_| draw(&mut state, 2) == 1)
            .map(|c| c as u32)
            .collect();
        let support = if support.is_empty() { vec![0] } else { support };
        let coeffs: Vec<Elem<F>> = if i % 2 == 0 {
            vec![Elem::<F>::ONE; support.len()]
        } else {
            support
                .iter()
                .map(|_| F::elem(&mut state))
                .map(|c| if c.is_zero() { Elem::<F>::ONE } else { c })
                .collect()
        };
        rhs.push(pack_row::<F>(&eval_eq::<F>(&support, &coeffs, &x)));
        rows.push((support, coeffs));
    }
    let mut hybrid = Hybrid::<F>::new(n, sym_len);
    for (i, (support, coeffs)) in rows.iter().enumerate() {
        if i % 2 == 0 {
            hybrid.push_binary_row(support, &rhs[i]);
        } else {
            hybrid.push_field_row(support, coeffs, &rhs[i]);
        }
    }
    let mut values = Matrix::<F>::zeros(n, sym_cols).unwrap();
    let mut determined = vec![false; n];
    let rank = hybrid.solve_into(&mut values, &mut determined).unwrap();
    let (rank_ref, consistent, sol_ref, determined_ref) =
        hybrid_reference::<F>(&rows, &rhs, n, sym_cols);
    assert!(
        consistent && rank_ref < n,
        "the case must be rank-deficient"
    );
    assert_eq!(rank, rank_ref, "rank against the dense reference");
    for (col, &expected) in determined_ref.iter().enumerate() {
        assert_eq!(determined[col], expected, "determinedness at col {col}");
        if expected {
            assert_eq!(values.row(col), sol_ref.as_ref().unwrap().row(col));
        }
    }
    let sol = naive_of::<F>(&values);
    assert_system_satisfied::<F>(&rows, &rhs, &sol);

    // An extra contradictory equation makes the system inconsistent.
    let (support, coeffs) = rows[0].clone();
    let mut bumped = eval_eq::<F>(&support, &coeffs, &sol);
    bumped[0] = bumped[0].add(Elem::<F>::ONE);
    let bumped = pack_row::<F>(&bumped);
    let mut hybrid = Hybrid::<F>::new(n, sym_len);
    for (i, (support, coeffs)) in rows.iter().enumerate() {
        hybrid.push_field_row(support, coeffs, &rhs[i]);
    }
    hybrid.push_field_row(&support, &coeffs, &bumped);
    let err = hybrid.solve_into(&mut values, &mut determined).unwrap_err();
    assert!(matches!(err, SolveError::Inconsistent { .. }));
    let (_, consistent_ref, _, _) = {
        let mut all_rows = rows.clone();
        all_rows.push((support, coeffs));
        let mut all_rhs = rhs.clone();
        all_rhs.push(bumped);
        hybrid_reference::<F>(&all_rows, &all_rhs, n, sym_cols)
    };
    assert!(!consistent_ref, "the reference must agree on inconsistency");
}

fn check_hybrid<F: PrimeField>() {
    // Symbol widths straddling the alignment boundary of the dense block.
    let base = ALIGN / F::BYTES;
    for sym_cols in [1, base - 1, base] {
        hybrid_consistent_case::<F>(12, sym_cols.max(1), &[]);
    }
    hybrid_consistent_case::<F>(14, 2, &[10, 11, 12, 13]);
    hybrid_deficient_and_inconsistent_case::<F>();
}

// ── Streaming echelon ────────────────────────────────────────────────────────

/// Absorbs a nonsingular system into a reduced echelon and checks the
/// recovered payloads, then probes the dependent and inconsistent verdicts,
/// the forward mode, the wide-coefficient path, and the unit shortcut.
fn check_echelon<F: PrimeField>() {
    for n in [1usize, 2, 3, 6, 12] {
        let sym_cols = 2;
        let sym_len = sym_bytes::<F>(sym_cols);
        let c = naive_invertible::<F>(n, 0xEC10 ^ n as u64);
        let x = naive_field::<F>(n, sym_cols, 0xEC20 ^ n as u64);
        let rhs = naive_mul::<F>(&c, &x);
        let mut ech = Echelon::<F>::new(n, sym_len, true).unwrap();
        for i in 0..n {
            let verdict = ech.absorb(&pack_row::<F>(&c[i]), &pack_row::<F>(&rhs[i]));
            assert!(
                matches!(verdict, Innovation::Innovative { .. }),
                "row {i} of n={n} must be innovative"
            );
            assert_eq!(ech.rank(), i + 1);
        }
        assert!(ech.is_complete());
        for (col, xv) in x.iter().enumerate() {
            assert_eq!(
                ech.recovered_value(col).unwrap(),
                pack_row::<F>(xv),
                "recovered value at n={n}, col={col}"
            );
        }
    }

    // A dependent absorption with a consistent payload, then an inconsistent
    // one.
    let n = 6;
    let sym_len = sym_bytes::<F>(1);
    let r0 = naive_field::<F>(1, n, 0xECA0);
    let r1 = naive_field::<F>(1, n, 0xECA1);
    let sum: Naive<F> = vec![r0[0].iter().zip(&r1[0]).map(|(a, b)| a.add(*b)).collect()];
    let zero = vec![0u8; sym_len];
    let mut ech = Echelon::<F>::new(n, sym_len, true).unwrap();
    ech.absorb(&pack_row::<F>(&r0[0]), &zero);
    ech.absorb(&pack_row::<F>(&r1[0]), &zero);
    assert!(matches!(
        ech.absorb(&pack_row::<F>(&sum[0]), &zero),
        Innovation::Dependent
    ));
    let one = pack_row::<F>(&[Elem::<F>::ONE]);
    assert!(matches!(
        ech.absorb(&pack_row::<F>(&sum[0]), &one),
        Innovation::Inconsistent
    ));

    // Forward mode: rank fills without recovery, and every retained row
    // still evaluates to its payload against the true solution.
    let n = 5;
    let sym_cols = 2;
    let sym_len = sym_bytes::<F>(sym_cols);
    let c = naive_invertible::<F>(n, 0xECB0);
    let x = naive_field::<F>(n, sym_cols, 0xECB1);
    let rhs = naive_mul::<F>(&c, &x);
    let mut ech = Echelon::<F>::new(n, sym_len, false).unwrap();
    for i in 0..n {
        assert!(matches!(
            ech.absorb(&pack_row::<F>(&c[i]), &pack_row::<F>(&rhs[i])),
            Innovation::Innovative { .. }
        ));
    }
    assert!(ech.is_complete());
    for row in ech.retained_rows() {
        let (head, tail) = row.coefficient_slices();
        let coeffs: Vec<Elem<F>> = head
            .chunks(F::BYTES)
            .chain(tail.chunks(F::BYTES))
            .map(F::decode)
            .collect();
        let mut acc = vec![Elem::<F>::ZERO; sym_cols];
        for (t, &c) in coeffs.iter().enumerate() {
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot = slot.add(c.mul(x[row.pivot() + t][k]));
            }
        }
        assert_eq!(
            pack_row::<F>(&acc),
            row.rhs(),
            "retained row {}",
            row.pivot()
        );
    }

    // Wide coefficient rows exercise the per-pivot right-hand-side path past
    // the stack gather window.
    let n = 70;
    let sym_len = sym_bytes::<F>(1);
    let c = naive_invertible::<F>(n, 0xECC0);
    let x = naive_field::<F>(n, 1, 0xECC1);
    let rhs = naive_mul::<F>(&c, &x);
    let mut ech = Echelon::<F>::new(n, sym_len, true).unwrap();
    for i in 0..n {
        ech.absorb(&pack_row::<F>(&c[i]), &pack_row::<F>(&rhs[i]));
    }
    assert!(ech.is_complete());
    for (col, xv) in x.iter().enumerate() {
        assert_eq!(ech.recovered_value(col).unwrap(), pack_row::<F>(xv));
    }

    // The systematic unit path.
    let mut ech = Echelon::<F>::new(4, sym_bytes::<F>(1), true).unwrap();
    let payload = pack_row::<F>(&naive_field::<F>(1, 1, 0xECD0)[0]);
    assert!(matches!(
        ech.absorb_unit(2, &payload),
        Innovation::Innovative { pivot: 2 }
    ));
    assert_eq!(ech.recovered_value(2).unwrap(), payload);
}

/// The whole public surface over one prime field.
fn run_all<F: PrimeField>() {
    check_ple::<F>();
    check_solve::<F>();
    check_inverse::<F>();
    check_det::<F>();
    check_gemm::<F>();
    check_trsm::<F>();
    check_small::<F>();
    check_hybrid::<F>();
    check_echelon::<F>();
}

#[test]
fn prime_field_operations_match_oracles() {
    run_all::<Mersenne31>();
    run_all::<Goldilocks>();
    run_all::<QuadMersenne31>();
}

#[test]
fn small_matrix_singular_error_reports_rank() {
    fn check<F: FieldBuffer>() {
        // [[0, 1], [0, 0]] has rank one; its first column has no pivot.
        let mut a = SmallMatrix::<F, 2>::zeros();
        a.set(0, 1, Elem::<F>::ONE);
        assert_eq!(a.rank(), 1);
        let rhs = Matrix::<F>::zeros(2, 1).unwrap();
        let mut out = Matrix::<F>::zeros(2, 1).unwrap();
        assert_eq!(
            a.solve_into(&rhs, &mut out),
            Err(SolveError::Singular { rank: 1, order: 2 })
        );
    }
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}
