//! Shared test helpers. The fixed-seed LCG noise convention mirrors `fgf`'s
//! kernel tests: deterministic bytes, no `rand`, same stream everywhere.
#![allow(dead_code)]

// Field type aliases. `fgf` 2.x exported these names at its root; 4.x spells
// each field as an explicit `FiniteField` instantiation.
pub type Gf8B = fgf::FiniteField<2, 8, fgf::binary::Polynomial<{ fgf::binary::AES }>>;
pub type Gf8D = fgf::FiniteField<2, 8, fgf::binary::Polynomial<{ fgf::binary::RS }>>;
pub type Gf16 = fgf::FiniteField<2, 16, fgf::extension::Extension<fgf::binary::Rijndael16, 2>>;
pub type Gf32 = fgf::FiniteField<2, 32, fgf::extension::Extension<fgf::binary::Rijndael32, 2>>;
pub type Gf64 = fgf::FiniteField<2, 64, fgf::extension::Extension<fgf::binary::Rijndael64, 2>>;
pub type FanPaar8 = fgf::FiniteField<2, 8, fgf::extension::Extension<fgf::binary::FanPaar8, 2>>;
pub type FanPaar16 = fgf::FiniteField<2, 16, fgf::extension::Extension<fgf::binary::FanPaar16, 2>>;
pub type FanPaar32 = fgf::FiniteField<2, 32, fgf::extension::Extension<fgf::binary::FanPaar32, 2>>;
pub type FanPaar64 = fgf::FiniteField<2, 64, fgf::extension::Extension<fgf::binary::FanPaar64, 2>>;
pub type Mersenne31 = fgf::prime::Mersenne31;
pub type Goldilocks = fgf::prime::Goldilocks;
pub type QuadMersenne31 = fgf::extension::QuadMersenne31;

/// `len` deterministic pseudo-random bytes from `seed`.
pub fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// The next LCG state, and a draw in `0..modulo` (`0` if `modulo == 0`).
pub fn draw(state: &mut u64, modulo: usize) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    if modulo == 0 {
        return 0;
    }
    ((*state >> 33) as usize) % modulo
}

/// Builds the `Perm` whose explicit image is `img`: applying it to `(0..n)`
/// yields `img`. `img` must be a permutation of `0..n`.
pub fn perm_from_image(img: &[usize]) -> gfm::Perm {
    let mut cur: Vec<usize> = (0..img.len()).collect();
    let mut p = gfm::Perm::identity(img.len());
    for (i, &want) in img.iter().enumerate() {
        let j = cur.iter().position(|&c| c == want).unwrap();
        p.record_swap(i, j);
        cur.swap(i, j);
    }
    p
}

/// Every permutation of `0..n` as explicit images, in lexicographic order.
pub fn all_images(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur: Vec<usize> = (0..n).collect();
    loop {
        out.push(cur.clone());
        // Next permutation in lexicographic order.
        let Some(i) = (0..cur.len().saturating_sub(1)).rfind(|&i| cur[i] < cur[i + 1]) else {
            break;
        };
        let j = (i + 1..cur.len()).rfind(|&j| cur[i] < cur[j]).unwrap();
        cur.swap(i, j);
        cur[i + 1..].reverse();
    }
    out
}

/// Dimension pairs covering degenerate shapes, lane boundaries, and a
/// deterministic pseudo-random spread.
pub fn sample_dims(seed: u64) -> Vec<(usize, usize)> {
    let mut dims = vec![
        (0, 0),
        (0, 7),
        (7, 0),
        (1, 1),
        (1, 32),
        (32, 1),
        (31, 33),
        (33, 31),
        (32, 32),
        (64, 64),
        (65, 63),
        (2, 2),
    ];
    let mut state = seed | 1;
    for _ in 0..8 {
        dims.push((draw(&mut state, 71), draw(&mut state, 71)));
    }
    dims
}

/// Column counts whose byte width (`cols * elem_bytes`) straddles `ALIGN`
/// boundaries, paired with a couple of small row counts.
pub fn straddling_dims(elem_bytes: usize, align: usize) -> Vec<(usize, usize)> {
    let mut dims = Vec::new();
    for k in 1..=2 {
        let base = align * k / elem_bytes;
        for cols in [base - 1, base, base + 1] {
            for rows in [1, 3] {
                dims.push((rows, cols));
            }
        }
    }
    dims
}
