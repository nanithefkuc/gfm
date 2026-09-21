//! The streaming accumulator: one packet in, an [`Innovation`] verdict out,
//! rank and recovered payloads maintained incrementally. This is on-the-fly
//! Gaussian elimination — the decode completes the instant rank fills, with no
//! batch phase — and it is the general form of the two bases `ccrlnc` keeps.

mod echelon;
mod innovation;

pub use echelon::{Echelon, RetainedRow};
pub use innovation::Innovation;
