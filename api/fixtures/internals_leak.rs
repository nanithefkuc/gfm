//! Feature-equality fixture: the internals feature also leaks a root item.

/// Present with and without the feature; a stable-path set must keep it.
pub fn internals_probe() {}

/// Present only with the feature; the stable export sets then differ.
#[cfg(feature = "internals")]
pub fn leaked() {}

#[cfg(feature = "internals")]
pub mod internals {
    /// Reachable only through the unstable facade.
    pub fn kernel() {}
}
