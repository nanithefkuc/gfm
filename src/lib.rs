//! gfm — Galois Field Math: matrices and linear systems over the finite
//! fields of `fgf`. Dense matrices hold over every field `fgf` packs into
//! byte lanes (`fgf::FieldBuffer`); bit-packed matrices cover GF(2).
//!
//! > `gfm` is a solver, not a codec. Field arithmetic and byte-buffer vector
//! > primitives come from `fgf` — never re-implement them here. Sparse graph
//! > topology, Tanner-graph generation, and peeling over a graph as symbols
//! > arrive belong to `fgraph`; `hybrid`'s weight-one pivot rule is an
//! > elimination order inside this crate's solve. Wire formats, shard
//! > ownership, rate adaptation, degree distributions, and codec shells
//! > belong to consumers. This crate receives a matrix and returns facts
//! > about it.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]
// Row and column counts arrive as `usize` and are routinely narrowed to
// `u32`/`u16` index types after geometry validation has bounded them; the
// truncation is checked at the validation boundary, not at every cast.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod bits;
mod blocking;
pub mod dense;
mod error;
pub mod hybrid;
pub mod incremental;
// Compiled in every feature set; public reachability comes from `internals`.
mod inspection;
#[cfg(feature = "internals")]
pub mod internals;
mod row_ops;

pub use error::{GeometryError, SolveError};
