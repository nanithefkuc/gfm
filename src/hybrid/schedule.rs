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

/// Given the weight-2 rows as edges `(col_a, col_b)` over active columns,
/// returns the index into `edges` of an edge lying in the largest connected
/// component (ties broken toward the earliest edge), or `None` if empty.
pub(crate) fn largest_component_edge(
    edges: &[(u32, u32)],
    cols: usize,
    parent: &mut Vec<usize>,
    rank: &mut Vec<u8>,
    size: &mut Vec<usize>,
) -> Option<usize> {
    if edges.is_empty() {
        return None;
    }
    parent.clear();
    parent.resize(cols, usize::MAX);
    rank.clear();
    rank.resize(cols, 0);
    size.clear();
    size.resize(cols, 0);
    for &(a, b) in edges {
        parent[a as usize] = a as usize;
        parent[b as usize] = b as usize;
    }

    for &(a, b) in edges {
        union(parent, rank, a as usize, b as usize);
    }
    for node in 0..cols {
        if parent[node] != usize::MAX {
            let root = find(parent, node);
            size[root] += 1;
        }
    }
    let mut best: Option<(usize, usize)> = None;
    for (idx, &(a, _)) in edges.iter().enumerate() {
        let root = find(parent, a as usize);
        let component_size = size[root];
        if best.is_none_or(|(best_size, _)| component_size > best_size) {
            best = Some((component_size, idx));
        }
    }
    best.map(|(_, idx)| idx)
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return;
    }
    match rank[ra].cmp(&rank[rb]) {
        core::cmp::Ordering::Less => parent[ra] = rb,
        core::cmp::Ordering::Greater => parent[rb] = ra,
        core::cmp::Ordering::Equal => {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }
}
