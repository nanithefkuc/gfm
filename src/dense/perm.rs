//! Permutations as LAPACK-style index vectors.
//!
//! A [`Perm`] on `n` positions is a vector of `n` entries where entry `i`
//! swaps positions `i` and `swap[i]`, with `swap[i] >= i`, applied in
//! increasing order — the same shape as LAPACK's `IPIV` and M4RIE's `mzp_t`.
//! Applying one to a slice is `n` in-place swaps: no allocation, no auxiliary
//! storage, and the data of a matrix never has to move for the elimination to
//! permute it.

use alloc::vec::Vec;

/// A permutation on `n` positions, stored as a transposition list.
///
/// Entry `i` holds the position exchanged with position `i` at step `i`;
/// entries with `swap[i] == i` are fixed points. Two `Perm`s are equal when
/// their transposition lists are identical; constructors in this crate
/// produce a canonical list for a given permutation, so structural equality
/// matches permutation equality.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Perm {
    swap: Vec<usize>,
}

impl Perm {
    /// The identity permutation on `n` positions.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            swap: (0..n).collect(),
        }
    }
    pub(crate) fn reset_identity(&mut self) {
        for (i, slot) in self.swap.iter_mut().enumerate() {
            *slot = i;
        }
    }

    /// The number of positions this permutation acts on.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.swap.len()
    }

    /// Whether this permutation acts on zero positions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.swap.is_empty()
    }

    /// Records an exchange of positions `i` and `j` at step `i`.
    ///
    /// `j == i` leaves position `i` fixed.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range, or if `j < i` (a LAPACK-style
    /// list swaps each position with itself or a later one).
    pub fn record_swap(&mut self, i: usize, j: usize) {
        assert!(i < self.swap.len() && i <= j && j < self.swap.len());
        self.swap[i] = j;
    }

    /// Applies the permutation to `v` in place: step `i` swaps `v[i]` and
    /// `v[swap[i]]` in increasing order.
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != self.len()`.
    pub fn apply<T>(&self, v: &mut [T]) {
        assert_eq!(v.len(), self.swap.len());
        for (i, &j) in self.swap.iter().enumerate() {
            v.swap(i, j);
        }
    }

    /// Applies the inverse permutation to `v` in place: the transpositions in
    /// reverse order. `apply_inv` after [`apply`](Self::apply) is the
    /// identity.
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != self.len()`.
    pub fn apply_inv<T>(&self, v: &mut [T]) {
        assert_eq!(v.len(), self.swap.len());
        for (i, &j) in self.swap.iter().enumerate().rev() {
            v.swap(i, j);
        }
    }

    /// The composition that applies `self` first and `then` second:
    /// `self.compose(then).apply(v)` equals `self.apply(v); then.apply(v)`.
    ///
    /// Composition is associative, and equal inputs produce byte-identical
    /// transposition lists.
    ///
    /// # Panics
    ///
    /// Panics if the two permutations differ in length.
    #[must_use]
    pub fn compose(&self, then: &Self) -> Self {
        assert_eq!(self.swap.len(), then.swap.len());
        let n = self.swap.len();
        // Explicit image of each: after applying `p` to (0..n), position `x`
        // holds the original index `img[x]`.
        let mut first: Vec<usize> = (0..n).collect();
        self.apply(&mut first);
        let mut second: Vec<usize> = (0..n).collect();
        then.apply(&mut second);
        // Composed image: `v` -> `self` -> `then` maps position `x` to the
        // original index `first[second[x]]`.
        let composed: Vec<usize> = second.iter().map(|&s| first[s]).collect();
        Self::from_image(&composed)
    }

    /// The parity of the permutation: `false` for even, `true` for odd.
    ///
    /// Every recorded transposition with distinct endpoints flips the parity,
    /// so this is the inversion count of the explicit permutation, mod 2 —
    /// the sign a determinant would multiply by outside characteristic two.
    #[must_use]
    pub fn parity(&self) -> bool {
        self.swap
            .iter()
            .enumerate()
            .filter(|&(i, &j)| i != j)
            .count()
            % 2
            == 1
    }

    /// The explicit image of one position: the original index whose content
    /// sits at `position` after applying this permutation. `O(n)`, no
    /// allocation: walks the swap list in reverse.
    ///
    /// # Panics
    ///
    /// Panics if `position >= self.len()`.
    #[must_use]
    pub fn image_at(&self, position: usize) -> usize {
        assert!(position < self.swap.len(), "position out of bounds");
        let mut pos = position;
        for (i, &j) in self.swap.iter().enumerate().rev() {
            if pos == i {
                pos = j;
            } else if pos == j {
                pos = i;
            }
        }
        pos
    }

    /// The raw swap list, for in-crate consumers that permute by hand.
    pub(crate) fn swaps(&self) -> &[usize] {
        &self.swap
    }

    /// The canonical transposition list for the explicit permutation `img`,
    /// where `img[x]` is the original index sitting at position `x` after
    /// application.
    fn from_image(img: &[usize]) -> Self {
        let n = img.len();
        let mut cur: Vec<usize> = (0..n).collect();
        let mut pos_of: Vec<usize> = (0..n).collect();
        let mut swap = Vec::with_capacity(n);
        for (i, &want) in img.iter().enumerate() {
            let j = pos_of[want];
            swap.push(j);
            cur.swap(i, j);
            pos_of.swap(cur[i], cur[j]);
        }
        Self { swap }
    }
}

/// Inverts a permutation of `0..n` in place, without allocating.
///
/// Values are tagged with a bitwise NOT while unprocessed, which is
/// distinguishable from any real value because every entry is `< n` (and `n`
/// is a row count, far below `usize::MAX / 2`). Each cycle is walked once,
/// writing `inv[p[cur]] = cur` at every step.
///
/// # Panics
///
/// Panics or corrupts silently if `p` is not a permutation of `0..p.len()`;
/// callers hold that invariant by construction.
pub(crate) fn invert_in_place(p: &mut [usize]) {
    let n = p.len();
    for v in p.iter_mut() {
        *v = !*v;
    }
    for i in 0..n {
        if p[i] >= n {
            let mut prev = i;
            let mut cur = !p[i];
            loop {
                let next = !p[cur];
                p[cur] = prev;
                if cur == i {
                    break;
                }
                prev = cur;
                cur = next;
            }
        }
    }
}
