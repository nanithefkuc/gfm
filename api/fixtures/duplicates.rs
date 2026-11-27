//! One-path checker fixture: every export shape the check must classify.

/// Direct membership plus a renamed re-export: two paths.
pub struct Direct;
pub use Direct as Renamed;

mod private {
    /// Re-exported from a private module: one path.
    pub struct FromPrivate;
}
pub use private::FromPrivate;

/// A re-exported module gives each descendant an alternate path.
pub mod outer {
    pub mod inner {
        pub struct Deep;
    }
}
pub use outer::inner as alias;

/// Duplicate exports of an external type.
pub use core::cell::Cell;
pub mod ext {
    pub use core::cell::Cell;
}

/// Documentation-hidden exports count like any other.
#[doc(hidden)]
pub struct Hidden;
pub mod hidden_again {
    #[doc(hidden)]
    pub use super::Hidden;
}

/// A stable item also exported through `internals`: one stable path.
pub struct Single;
pub mod internals {
    pub use super::Single;
}

/// A named type alias is a distinct item.
pub type Alias = Single;

/// A glob re-export is a violation and duplicates what it imports.
pub mod globbed {
    pub struct Globbed;
}
pub mod regathered {
    pub use super::globbed::*;
}
