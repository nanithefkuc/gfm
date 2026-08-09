/* C shim over M4RIE, for the GF(2^8) `mzed_ple` differential. The field is
 * built with fgf's AES reduction polynomial `0x11B`, so the byte encoding of
 * an element matches `fgf::Gf8` exactly and the rank is a like-for-like
 * comparison. The element setters are `static inline` in the headers, so the
 * bridge must be compiled against the installed M4RIE. */

#include <m4rie/m4rie.h>
#include <stddef.h>
#include <stdint.h>

/* Rank of a `rows × cols` GF(2^8) matrix given as `rows * cols` bytes
 * (row-major, one polynomial-coefficient byte per element), via `mzed_ple`
 * under the `0x11B` field. */
size_t gfm_m4rie_gf8_rank(const uint8_t *data, size_t rows, size_t cols) {
    gf2e *ff = gf2e_init(0x11B);
    mzed_t *a = mzed_init(ff, (rci_t)rows, (rci_t)cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            mzed_write_elem(a, (rci_t)r, (rci_t)c, (word)data[r * cols + c]);
        }
    }
    mzp_t *p = mzp_init(a->nrows);
    mzp_t *q = mzp_init(a->ncols);
    rci_t rank = mzed_ple(a, p, q);
    mzp_free(p);
    mzp_free(q);
    mzed_free(a);
    gf2e_free(ff);
    return (size_t)rank;
}
