//! The result of absorbing one equation into an [`Echelon`](super::Echelon).

/// What one absorbed row did to an accumulator.
///
/// `I` names the pivot coordinate: [`Echelon`](super::Echelon) uses a relative
/// `usize` column, while a sliding-window caller can map it to an absolute
/// sequence coordinate without defining a second verdict type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Innovation<I = usize> {
    /// The row was independent of the retained rows and became a new pivot at
    /// the named coordinate.
    Innovative {
        /// The pivot coordinate the new row occupies.
        pivot: I,
    },
    /// The row was a linear combination of retained rows; its right-hand side
    /// was consistent and nothing changed.
    Dependent,
    /// The coefficients were dependent but the right-hand side contradicted
    /// the retained rows — the system as posed has no solution.
    Inconsistent,
}
