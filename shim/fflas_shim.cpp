/* C++ shim over FFLAS-FFPACK, for a GF(2) rank differential. GF(2) is the
 * prime field FFPACK handles natively through `Givaro::Modular`, so the
 * element encoding is a trivial 0/1 integer and the comparison against
 * `bits::Ple` is exact. Compiled against the installed FFLAS-FFPACK by
 * `build.rs`; the templates force a C++ translation unit behind a plain C
 * entry point. */

#include <cstddef>
#include <cstdint>

#include <fflas-ffpack/fflas-ffpack.h>
#include <givaro/modular.h>

extern "C" size_t gfm_ffpack_gf2_rank(const uint8_t *data, size_t m,
                                      size_t n) {
    typedef Givaro::Modular<int64_t> Field;
    Field field(2);
    Field::Element_ptr a = FFLAS::fflas_new(field, m, n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            field.init(a[i * n + j], static_cast<int64_t>(data[i * n + j]));
        }
    }
    size_t rank = FFPACK::Rank(field, m, n, a, n);
    FFLAS::fflas_delete(a);
    return rank;
}
