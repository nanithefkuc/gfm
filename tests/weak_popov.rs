//! `weak_popov_with_scratch` reuses its per-column collision buffer, so
//! reducing a stream of equally shaped bases reaches a zero-allocation steady
//! state. Proven under a counting global allocator with a row type whose
//! coefficient storage is pre-sized, isolating the reducer's own allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::Gf8;
use fgf::field::{Elem, Field};
use gfm::{ReduceError, WeakPopovRow, WeakPopovScratch, weak_popov, weak_popov_with_scratch};

/// A global allocator that counts calls to `alloc`.
struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

type E = <Gf8 as Field>::Elem;

/// A weak-Popov row whose columns are pre-sized, so no reduction step grows or
/// allocates its coefficient storage.
struct FixedRow {
    columns: Vec<Vec<E>>,
    capacity: usize,
}

impl FixedRow {
    fn new(columns: &[&[u8]], capacity: usize) -> Self {
        let columns = columns
            .iter()
            .map(|column| {
                let mut buffer = Vec::with_capacity(capacity);
                buffer.extend(column.iter().map(|&value| E::from_raw(value)));
                buffer
            })
            .collect();
        Self { columns, capacity }
    }

    fn leading_columns(basis: &[FixedRow], shifts: &[usize]) -> Vec<usize> {
        basis
            .iter()
            .filter_map(|row| row.leading_term(shifts).unwrap())
            .map(|term| term.column)
            .collect()
    }
}

impl WeakPopovRow<Gf8> for FixedRow {
    type Error = ReduceError;

    fn column_count(&self) -> usize {
        self.columns.len()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.columns[column]
            .iter()
            .rposition(|coefficient| !coefficient.is_zero())
    }

    fn coefficient(&self, column: usize, degree: usize) -> E {
        self.columns[column].get(degree).copied().unwrap_or(E::ZERO)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: E,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        for (target, source) in self.columns.iter_mut().zip(&pivot.columns) {
            let required = source.len() + shift;
            assert!(
                required <= self.capacity,
                "test row exceeded its pre-sized capacity"
            );
            if required > target.len() {
                target.resize(required, E::ZERO);
            }
            for (degree, &coefficient) in source.iter().enumerate() {
                let position = degree + shift;
                target[position] = target[position].add(scale.mul(coefficient));
            }
            while target
                .last()
                .is_some_and(|coefficient| coefficient.is_zero())
            {
                target.pop();
            }
        }
        Ok(())
    }
}

fn basis(first: &[&[u8]], second: &[&[u8]], capacity: usize) -> [FixedRow; 2] {
    [
        FixedRow::new(first, capacity),
        FixedRow::new(second, capacity),
    ]
}

#[test]
fn scratch_reuse_reduces_without_allocating() {
    let capacity = 4;
    let shifts = [0usize, 0usize];
    let mut scratch = WeakPopovScratch::new();

    // Warm-up: the first reduction may allocate the scratch's column buffer.
    let mut warm = basis(&[&[0, 1], &[1]], &[&[1], &[]], capacity);
    weak_popov_with_scratch::<Gf8, _>(&mut warm, &shifts, &mut scratch).unwrap();

    // Build the measured inputs outside the counted window.
    let mut first = basis(&[&[0, 1], &[1]], &[&[1], &[]], capacity);
    let mut second = basis(&[&[1, 1], &[1]], &[&[0, 1], &[1]], capacity);

    let before = ALLOCATIONS.load(Ordering::SeqCst);
    weak_popov_with_scratch::<Gf8, _>(black_box(&mut first), &shifts, &mut scratch).unwrap();
    weak_popov_with_scratch::<Gf8, _>(black_box(&mut second), &shifts, &mut scratch).unwrap();
    let allocations = ALLOCATIONS.load(Ordering::SeqCst) - before;

    assert_eq!(
        allocations, 0,
        "a warmed weak_popov_with_scratch reduction allocated"
    );

    // The reused-scratch result matches the standalone entry point exactly.
    let mut plain = basis(&[&[1, 1], &[1]], &[&[0, 1], &[1]], capacity);
    weak_popov::<Gf8, _>(&mut plain, &shifts).unwrap();
    assert_eq!(
        FixedRow::leading_columns(&second, &shifts),
        FixedRow::leading_columns(&plain, &shifts),
    );
}
