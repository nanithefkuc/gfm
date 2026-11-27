//! The hybrid sparse→dense inactivation solver.
//!
//! A caller pushes sparse equations (binary or field-valued) with packed
//! symbol payloads; [`Hybrid::solve`] peels the sparse structure, inactivates
//! the columns that would otherwise cause fill, solves the small dense block
//! with [`crate::dense::Ple`], and back-substitutes. Inactivation is an
//! ordering, not a different computation: rank, the consistency verdict,
//! which columns are determined, and their values agree with a full dense
//! solve of the same system. Undetermined columns take a schedule-dependent
//! particular solution; they need not be zero or match any other solve.
//!
//! Payloads accept every lane the field decoder accepts; verification
//! compares decoded values, and solution values may keep noncanonical input
//! bytes.

mod colindex;
mod deferred;
mod schedule;
mod solve;
mod source;
mod sparse;

pub use solve::{Hybrid, Solution, SolveStats};
pub use source::{DenseRow, DenseRows};
