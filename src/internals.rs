//! Implementation details for benchmarking and experimentation, with no
//! compatibility guarantees.
//!
//! Paths, signatures, and behavior may change or disappear in any release.
//! Production consumers use the supported API at the crate root and in
//! [`crate::bits`] and [`crate::dense`]. Enabling this feature changes what
//! is reachable, never what is compiled: every item below is part of the
//! ordinary build.

pub use crate::internal_api::{
    BitMatrixInternals, BitPleInternals, HybridInternals, MatrixInternals, PleInternals, row_ops,
};
