/* C shim over M4RI: the bit setters and readers are `static inline` in the
 * headers and export no linkable symbol, so a differential test cannot FFI
 * to them directly. This wrapper is compiled against the installed M4RI when
 * `build.rs` finds it, and exposes one plain C entry point. */

#include <m4ri/m4ri.h>
#include <stddef.h>
#include <stdint.h>

/* Rank of a `rows × cols` GF(2) matrix given as `rows * words` packed u64
 * words (row-major, `words = ceil(cols / 64)`), via full echelonization. */
size_t gfm_m4ri_rank(const uint64_t *packed, size_t rows, size_t cols,
                     size_t words) {
    mzd_t *m = mzd_init((rci_t)rows, (rci_t)cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            uint64_t w = packed[r * words + (c >> 6)];
            if ((w >> (c & 63)) & 1u) {
                mzd_write_bit(m, (rci_t)r, (rci_t)c, 1);
            }
        }
    }
    rci_t rank = mzd_echelonize(m, 1);
    mzd_free(m);
    return (size_t)rank;
}
