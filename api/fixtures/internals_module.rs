//! Feature-equality fixture: the internals feature adds only the facade.

/// Present with and without the feature; a stable-path set must keep it.
pub fn internals_probe() {}

#[cfg(feature = "internals")]
pub mod internals {
    /// Reachable only through the unstable facade.
    pub fn kernel() {}
}
