//! The GF(2) storage domain: `u64`-packed bit matrices. Shares naming, error
//! types, and the permutation contract with [`crate::dense`]; shares no code.

mod derive;
mod matrix;
pub(crate) mod ple;

pub use derive::SolveScratch;
pub use matrix::{ALIGN, BitMatrix};
pub use ple::{Ple, PleScratch};
