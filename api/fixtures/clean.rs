//! One-path checker fixture with no violations.

mod private {
    pub struct FromPrivate;
}
pub use private::FromPrivate;

pub struct Single;
pub type Alias = Single;

pub mod internals {
    pub use super::Single;
    pub use super::private::*;
}
