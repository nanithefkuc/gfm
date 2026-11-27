//! The streaming accumulator: one packet in, an [`Innovation`] verdict out,
//! rank and recovered payloads maintained incrementally. This is on-the-fly
//! Gaussian elimination — the decode completes the instant rank fills, with no
//! batch phase — and it is the general form of the two bases `ccrlnc` keeps.
//!
//! [`EchelonMode`] selects forward-echelon retention for recoding or
//! recovered-unit propagation for decoding. Propagation substitutes recovered
//! unit rows and reports a sound subset of the determined coordinates; it is
//! not reduced row echelon form, and a determined coordinate whose retained
//! row still carries free coordinates is not reported. An exact route such as
//! [`crate::Ple::rref_into`] reports every determined coordinate.
//!
//! Packed inputs accept every lane the field decoder accepts; dependence and
//! consistency are decided on decoded values, and retained rows and recovered
//! payloads may keep noncanonical input bytes.

mod echelon;
mod innovation;

pub use echelon::{Echelon, EchelonMode, RetainedRow};
pub use innovation::Innovation;
