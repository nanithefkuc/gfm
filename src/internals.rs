//! Implementation details for benchmarking and experimentation, with no
//! compatibility guarantees.
//!
//! Paths, signatures, and behavior may change or disappear in any release.
//! Production consumers use the supported API in [`crate::bits`],
//! [`crate::dense`], [`crate::hybrid`], and [`crate::incremental`]. Enabling
//! this feature changes what is reachable, never what is compiled: every item
//! below is part of the ordinary build.

pub use crate::hybrid::solve::SolveStats;
pub use crate::inspection::{
    BitMatrixInternals, BitPleInternals, HybridInternals, MatrixInternals, PleInternals, row_ops,
};
