//! The column-to-row index of the sparse phase, in one arena.
//!
//! A pivot's elimination visits exactly the rows listed under its column, so
//! the index is read once per pivot and appended to whenever a merge widens a
//! destination's support. One `Vec<u32>` per column costs an allocation per
//! column at setup — `n` of them for a solver the caller builds fresh per
//! system, which is what a codec does — plus a growth realloc for most of
//! them and a pointer chase per probe.
//!
//! Here every column's list is a span of one flat buffer: a capacity is
//! reserved from the row supports the index is built over, and a column that
//! outgrows it moves to a fresh doubled span at the end of the buffer.
//! Appends are rare (a merge that adds an active column), so the moves are
//! rare, and the buffer only ever grows.

use alloc::vec::Vec;

pub(crate) struct ColumnIndex {
    /// Start of each column's span in `data`.
    start: Vec<u32>,
    /// Live entries in each column's span.
    len: Vec<u32>,
    /// Capacity of each column's span.
    cap: Vec<u32>,
    data: Vec<u32>,
}

impl ColumnIndex {
    pub(crate) const fn new() -> Self {
        Self {
            start: Vec::new(),
            len: Vec::new(),
            cap: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Whether a laid-out span table already covers `cols` columns, so a
    /// rebuild over unchanged supports can skip counting and layout.
    pub(crate) fn covers(&self, cols: usize) -> bool {
        self.start.len() == cols
    }

    /// Empties every column's list, keeping the span layout.
    pub(crate) fn clear(&mut self) {
        self.len.fill(0);
    }

    /// Starts a rebuild over `cols` columns: every column's entry count is
    /// zeroed, ready for [`Self::count`].
    pub(crate) fn begin(&mut self, cols: usize) {
        self.len.clear();
        self.len.resize(cols, 0);
    }

    /// Counts one entry for `column` during the first build pass.
    pub(crate) fn count(&mut self, column: usize) {
        self.len[column] += 1;
    }

    /// Lays out one span per column from the counted entries, with a
    /// quarter of slack (rounded up) so the common merge-time append lands
    /// in place, and reopens every span empty for [`Self::place`].
    ///
    /// The buffer is grown, never re-zeroed: a span's entries are written
    /// before they can be read.
    pub(crate) fn finish_counts(&mut self) {
        let cols = self.len.len();
        self.start.clear();
        self.start.reserve(cols);
        self.cap.clear();
        self.cap.reserve(cols);
        let mut offset = 0u32;
        for column in 0..cols {
            let count = self.len[column];
            let capacity = count + count / 4 + 1;
            self.start.push(offset);
            self.cap.push(capacity);
            offset += capacity;
        }
        if self.data.len() < offset as usize {
            self.data.resize(offset as usize, 0);
        }
        self.clear();
    }

    /// Writes one counted entry during the second build pass.
    pub(crate) fn place(&mut self, column: usize, row: u32) {
        let at = self.start[column] + self.len[column];
        self.data[at as usize] = row;
        self.len[column] += 1;
    }
    /// The number of rows listed under `column`.
    pub(crate) fn len(&self, column: usize) -> usize {
        self.len[column] as usize
    }

    /// The `index`-th row listed under `column`.
    pub(crate) fn entry(&self, column: usize, index: usize) -> u32 {
        self.data[self.start[column] as usize + index]
    }

    /// Lists `row` under `column`, moving the column's span to a fresh
    /// doubled one at the end of the buffer if it is full.
    pub(crate) fn push(&mut self, column: usize, row: u32) {
        let (start, len, cap) = (
            self.start[column] as usize,
            self.len[column] as usize,
            self.cap[column] as usize,
        );
        if len == cap {
            let grown = (cap * 2).max(4);
            let moved = self.data.len() as u32;
            self.data.reserve(grown);
            for index in 0..len {
                let value = self.data[start + index];
                self.data.push(value);
            }
            self.data.resize(moved as usize + grown, 0);
            self.start[column] = moved;
            self.cap[column] = grown as u32;
        }
        let at = self.start[column] as usize + len;
        self.data[at] = row;
        self.len[column] += 1;
    }
}
