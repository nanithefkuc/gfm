//! Pivot-schedule helpers. Correctness does not depend on the schedule — any
//! pivot order performs exact elimination — but the schedule decides how many
//! columns get inactivated, and that decides the size of the dense block.
//!
//! The one graph algorithm in the crate lives here: when the lightest active
//! rows have weight two, RFC 6330 §5.4.2.2 breaks the tie by the *largest
//! connected component* of the graph whose nodes are active columns and whose
//! edges are the weight-2 rows. Pivoting inside the biggest component keeps the
//! peel going longest and holds the inactivation count down.

use alloc::vec::Vec;

/// `parent` entry of a column outside the current edge set.
const UNVISITED: u32 = u32::MAX;

/// Reusable union-find state for the weight-two tie-break.
///
/// The arrays are column-indexed and sized once, but a query only ever
/// touches the endpoints of its own edge set: entries are returned to
/// [`UNVISITED`] on the way out, so membership is the parent slot itself and
/// no per-query sweep of the column space is needed. Component sizes ride
/// along in the union, which removes the second pass over the endpoints, and
/// the winning edge is the first to attain the maximum size, which lets the
/// third pass stop early. The query runs once per weight-two pivot over an
/// edge set that is a small fraction of the columns, so per-query work is
/// what matters and per-solve work is what does not exist.
pub(crate) struct Components {
    /// Union-find parent, [`UNVISITED`] outside the current edge set.
    parent: Vec<u32>,
    /// Node count of the component rooted here; meaningful at roots only.
    size: Vec<u32>,
    /// Distinct endpoints of the current edge set.
    touched: Vec<u32>,
}

impl Components {
    pub(crate) const fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
            touched: Vec::new(),
        }
    }

    /// Given the weight-2 rows as edges `(col_a, col_b)` over active columns,
    /// returns the index into `edges` of an edge lying in the largest
    /// connected component (ties broken toward the earliest edge), or `None`
    /// if empty.
    pub(crate) fn largest_component_edge(
        &mut self,
        edges: &[(u32, u32)],
        cols: usize,
    ) -> Option<usize> {
        if edges.is_empty() {
            return None;
        }
        if self.parent.len() < cols {
            self.parent.resize(cols, UNVISITED);
            self.size.resize(cols, 0);
        }
        self.touched.clear();
        for &(a, b) in edges {
            for node in [a as usize, b as usize] {
                if self.parent[node] == UNVISITED {
                    self.parent[node] = node as u32;
                    self.size[node] = 1;
                    self.touched.push(node as u32);
                }
            }
        }

        let mut max_size = 1u32;
        for &(a, b) in edges {
            let merged = union(&mut self.parent, &mut self.size, a, b);
            max_size = max_size.max(merged);
        }
        // The original scan keeps the first edge of strictly greatest
        // component size, which is the first edge attaining the maximum.
        let mut best = 0usize;
        for (index, &(a, _)) in edges.iter().enumerate() {
            let root = find(&mut self.parent, a) as usize;
            if self.size[root] == max_size {
                best = index;
                break;
            }
        }
        for &node in &self.touched {
            self.parent[node as usize] = UNVISITED;
        }
        Some(best)
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let grandparent = parent[parent[x as usize] as usize];
        parent[x as usize] = grandparent;
        x = grandparent;
    }
    x
}

/// Merges the components of `a` and `b`, returning the size of the component
/// they end up in.
fn union(parent: &mut [u32], size: &mut [u32], a: u32, b: u32) -> u32 {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return size[ra as usize];
    }
    let (ra, rb) = (ra as usize, rb as usize);
    // Union by size: the partition and every component size are independent
    // of which root wins, and the smaller tree hanging under the larger keeps
    // find paths short without a separate rank array.
    let (big, small) = if size[ra] >= size[rb] {
        (ra, rb)
    } else {
        (rb, ra)
    };
    parent[small] = big as u32;
    size[big] += size[small];
    size[big]
}
