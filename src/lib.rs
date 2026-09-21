//! gfm — Galois Field Math: dense, bit-packed, and hybrid sparse/dense
//! linear algebra over GF(2) and GF(2^m).
//!
//! > `gfm` is a solver, not a codec. Field arithmetic and byte-buffer vector
//! > primitives come from `fgf` — never re-implement them here. Sparse graph
//! > topology, Tanner-graph generation, and peeling belong to `fgraph`. Wire
//! > formats, shard ownership, rate adaptation, degree distributions, and
//! > codec shells belong to consumers. This crate receives a matrix and
//! > returns facts about it.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]
// Row and column counts arrive as `usize` and are routinely narrowed to
// `u32`/`u16` index types after geometry validation has bounded them; the
// truncation is checked at the validation boundary, not at every cast.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod backend;
pub mod bits;
pub mod dense;
mod error;
pub mod hybrid;
pub mod incremental;
// Compiled in every feature set; public reachability comes from `internals`.
#[allow(dead_code)]
mod internal_api;
#[cfg(feature = "internals")]
pub mod internals;
mod row_ops;

pub use backend::{Backend, backend_for};
pub use bits::BitMatrix;
pub use dense::{
    Matrix, Perm, Ple, PleScratch, SmallMatrix, SolveScratch, View, ViewMut, mul_add, mul_into,
    solve_lower_unit_assign, solve_upper_assign,
};
pub use error::{GeometryError, SolveError};
pub use hybrid::{DenseRow, DenseRows, Hybrid, Solution, SolveStats};
pub use incremental::{Echelon, Innovation, RetainedRow};
