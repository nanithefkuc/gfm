//! The GF(2) storage domain: bit-packed matrices in [`fgf::bits`] layout.
//! Shares naming, error types, and the permutation contract with
//! [`crate::dense`]; shares no code. Row-vector arithmetic — XORs, masked
//! ranges, weights — is [`fgf::bits`]'s; this module owns geometry,
//! indexing, and the single elimination.

mod derive;
mod matrix;
pub(crate) mod ple;

pub use derive::SolveScratch;
pub use matrix::{ALIGN, BitMatrix};
pub use ple::{Ple, PleScratch};
