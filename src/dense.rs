//! The packed-element storage domain: pitched, aligned, zero-padded matrices
//! over any `fgf` field, the permutation type, and the elimination that
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
pub use gemm::{mul_add, mul_into};
pub use matrix::{Matrix, ViewMut};
pub use perm::Perm;
pub(crate) use perm::invert_in_place;
pub use ple::Ple;
pub use small::SmallMatrix;
pub use trsm::{solve_lower_unit_assign, solve_upper_assign};
