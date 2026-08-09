//! The hybrid sparse→dense inactivation solver — the crate's differentiator.
//!
//! A caller pushes sparse equations (binary or field-valued) with packed
//! symbol payloads; [`Hybrid::solve`] peels the sparse structure, inactivates
//! the columns that would otherwise cause fill, solves the small dense block
//! with [`crate::dense::Ple`], and back-substitutes. The answer is identical
//! to a full dense solve over the same system — inactivation is an ordering,
//! not a different computation.

mod deferred;
mod schedule;
mod solve;
mod source;
mod sparse;

pub use solve::{Hybrid, Solution, SolveStats};
pub use source::{DenseRow, DenseRows};
