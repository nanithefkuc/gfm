//! Structured matrices with closed-form inverses: the MDS constructions FEC
//! actually uses, plus the batch inversion they lean on. Each avoids general
//! elimination by exploiting its structure — `O(k²)` where `Ple` would be
//! `O(k³)`.

mod batch_inv;
mod cauchy;
mod cauchy_inv;
mod vandermonde;

pub use batch_inv::batch_invert;
pub use cauchy::Cauchy;
pub use cauchy_inv::{cauchy_inverse_coefficients_into, cauchy_scratch_len};
pub use vandermonde::Vandermonde;
