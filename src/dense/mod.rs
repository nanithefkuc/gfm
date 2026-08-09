//! The GF(2^m) storage domain: pitched, aligned, zero-padded matrices of
//! packed field elements, the permutation type, and the elimination that
//! everything derives from.

mod derive;
mod gemm;
pub mod layout;
mod matrix;
mod perm;
mod ple;
mod small;
mod trsm;

pub use derive::SolveScratch;
pub use gemm::{mul_add_into, mul_into};
pub use matrix::{Matrix, View, ViewMut};
pub use perm::Perm;
pub(crate) use perm::invert_in_place;
pub use ple::{Ple, PleScratch};
pub use small::SmallMatrix;
pub use trsm::{solve_lower_unit_into, solve_upper_into};
