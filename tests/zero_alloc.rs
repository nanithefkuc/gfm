//! Steady-state container operations perform no allocation, proven under a
//! counting global allocator. Setup allocates; the measured section — views,
//! row access, swaps, permutations, and the one physical compaction — must
//! not.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use common::noise;
use fgf::Gf8;
use gfm::bits::{Ple as BitPle, PleScratch as BitPleScratch, SolveScratch as BitSolveScratch};
use gfm::{BitMatrix, Echelon, Hybrid, Matrix, Perm, Ple, PleScratch, SolveScratch};

/// A global allocator that counts calls to `alloc` on the current thread.
struct CountingAllocator;

thread_local! {
    // Per-thread tally: libtest runs each `#[test]` on its own worker thread,
    // so a process-global counter would also see the harness thread's
    // allocations and flake. A const initializer keeps this off the lazy-init
    // path, so touching it inside the allocator hook never re-enters `alloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// The current thread's allocation tally since process start.
fn alloc_count() -> usize {
    ALLOCATIONS.with(Cell::get)
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// One test, one process-wide counter: two parallel tests would race on it.
#[test]
fn steady_state_ops_do_not_allocate() {
    // Setup: allocation is expected and allowed here.
    let rows = 37;
    let mut m = Matrix::<Gf8>::zeros(rows, 100).unwrap();
    for r in 0..rows {
        m.row_mut(r).copy_from_slice(&noise(100, r as u64 + 1));
    }
    let mut p = Perm::identity(rows);
    for i in 0..rows {
        p.record_swap(i, i + (rows - i) / 2);
    }
    let mut bm = BitMatrix::zeros(rows, 100).unwrap();
    for r in 0..rows {
        for c in 0..100 {
            bm.set(r, c, (r * 100 + c).is_multiple_of(3));
        }
    }
    let mut probe: Vec<usize> = (0..rows).collect();

    let start = alloc_count();

    // The measured section: every non-constructor container operation.
    {
        let mut v = m.as_view_mut();
        v.swap_rows(3, 11);
        v.set(5, 6, fgf::gf8::Elem(0x5A));
        black_box(v.get(5, 6));
        black_box(v.row(7));
        black_box(v.row_mut(8));
        let view = v.as_view();
        let (top, bot) = view.split_rows(9).unwrap();
        black_box(top.row(2));
        black_box(bot.row(3));
        black_box(top.split_rows(4));
    }
    m.swap_rows(0, 36);
    m.apply_row_perm(&p).unwrap();
    m.compact_rows();
    p.apply(&mut probe);
    p.apply_inv(&mut probe);
    black_box(p.parity());
    black_box(p.len());
    bm.swap_rows(1, 35);
    bm.apply_row_perm(&p).unwrap();
    bm.compact_rows();
    bm.set(2, 99, true);
    black_box(bm.get(2, 99));
    black_box(bm.row(4));

    assert_eq!(
        alloc_count() - start,
        0,
        "steady-state container operations allocated",
    );

    // Second section: the derived queries. Construction allocates; the
    // measured section must not.
    let n = 12;
    let mut m = Matrix::<Gf8>::zeros(n, n).unwrap();
    for r in 0..n {
        m.row_mut(r).copy_from_slice(&noise(n, r as u64 + 3));
    }
    let ple = Ple::decompose(m, &mut PleScratch::new());
    let rank = ple.rank();
    let mut rref_out = Matrix::<Gf8>::zeros(n, n).unwrap();
    let mut kernel_out = Matrix::<Gf8>::zeros(n, n - rank).unwrap();
    let mut inv_out = Matrix::<Gf8>::zeros(n, n).unwrap();
    let rhs_data = noise(n * 2, 0xCAFE);
    let rhs = Matrix::<Gf8>::from_rows(n, 2, &rhs_data).unwrap();
    let mut sol = Matrix::<Gf8>::zeros(n, 2).unwrap();
    let mut solve_scratch = SolveScratch::new();
    // Warm the scratch: the first solve sizes its workspace.
    let _ = ple.solve_into(&rhs, &mut sol, &mut solve_scratch);

    let start = alloc_count();

    ple.rref_into(&mut rref_out);
    ple.kernel_into(&mut kernel_out);
    black_box(ple.det());
    black_box(ple.row_rank_profile());
    black_box(ple.col_rank_profile());
    let _ = ple.inverse_into(&mut inv_out);
    let _ = ple.solve_into(&rhs, &mut sol, &mut solve_scratch);

    assert_eq!(
        alloc_count() - start,
        0,
        "derived queries allocated in steady state",
    );

    // Third section: the bit domain's derived queries. Construction
    // allocates; every `*_into` form must not in steady state.
    let n = 12;
    let mut bm = BitMatrix::zeros(n, n).unwrap();
    for r in 0..n {
        for c in 0..n {
            // Unit upper triangular: full rank, so the inverse path runs.
            if r == c || (c > r && (r * n + c).is_multiple_of(2)) {
                bm.set(r, c, true);
            }
        }
    }
    let bple = BitPle::decompose(bm, &mut BitPleScratch::new());
    let brank = bple.rank();
    let mut brref = BitMatrix::zeros(n, n).unwrap();
    let mut bkernel = BitMatrix::zeros(n, n - brank).unwrap();
    let mut binv = BitMatrix::zeros(n, n).unwrap();
    let mut brhs = BitMatrix::zeros(n, 2).unwrap();
    for r in 0..n {
        brhs.set(r, 0, r.is_multiple_of(2));
        brhs.set(r, 1, true);
    }
    let mut bsol = BitMatrix::zeros(n, 2).unwrap();
    let mut bscratch = BitSolveScratch::new();
    // Warm the scratch: the first solve sizes its workspace.
    let _ = bple.solve_into(&brhs, &mut bsol, &mut bscratch);

    let start = alloc_count();

    bple.rref_into(&mut brref);
    bple.kernel_into(&mut bkernel);
    black_box(bple.det());
    black_box(bple.row_rank_profile());
    black_box(bple.col_rank_profile());
    let _ = bple.inverse_into(&mut binv);
    let _ = bple.solve_into(&brhs, &mut bsol, &mut bscratch);

    assert_eq!(
        alloc_count() - start,
        0,
        "bit-domain derived queries allocated in steady state",
    );

    // Fourth section: the streaming accumulator. Construction allocates its
    // state and scratch once; steady-state absorb must not allocate.
    let cols = 24;
    let s = 4;
    let mut ech = Echelon::<Gf8>::new(cols, s, true).unwrap();
    // Wire rows: coefficients and payload as packed bytes.
    let rows: Vec<Vec<u8>> = (0..cols).map(|r| noise(cols, 0x700 + r as u64)).collect();
    let payloads: Vec<Vec<u8>> = (0..cols).map(|r| noise(s, 0x900 + r as u64)).collect();
    // Warm: absorb the first row so any first-touch settling is outside the
    // measured window (there is none, but this matches the other sections).
    ech.absorb(&rows[0], &payloads[0]);

    let start = alloc_count();

    for r in 1..cols {
        black_box(ech.absorb(&rows[r], &payloads[r]));
    }
    for pair in ech.recovered() {
        black_box(pair);
    }

    assert_eq!(
        alloc_count() - start,
        0,
        "streaming absorb allocated in steady state",
    );
    // Fifth section: the complete sparse→dense solve. The first call sizes
    // every sparse, schedule, decomposition, and solution workspace.
    let cols = 64;
    let symbol_bytes = 32;
    let mut hybrid = Hybrid::<Gf8>::new(cols, symbol_bytes);
    let zero = vec![0; symbol_bytes];
    for column in 0..cols {
        hybrid.push_binary_row(&[column as u32], &zero);
    }
    for column in 0..cols {
        let mut pair = [column as u32, ((column + 1) % cols) as u32];
        pair.sort_unstable();
        hybrid.push_binary_row(&pair, &zero);
    }
    let mut hybrid_values = Matrix::<Gf8>::zeros(cols, symbol_bytes).unwrap();
    let mut hybrid_determined = vec![false; cols];
    hybrid
        .solve_into(&mut hybrid_values, &mut hybrid_determined)
        .unwrap();

    let start = alloc_count();
    black_box(
        hybrid
            .solve_into(&mut hybrid_values, &mut hybrid_determined)
            .unwrap(),
    );
    assert_eq!(
        alloc_count() - start,
        0,
        "hybrid solve allocated in steady state",
    );
}
