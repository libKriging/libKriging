#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LKALLOC_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LKALLOC_HPP

#if defined(ARMA_INCLUDES)
#error "<armadillo> should not be included before lkalloc.hpp"
#endif

#include <cstddef>

#include "libKriging/libKriging_exports.h"

namespace lkalloc {
LIBKRIGING_EXPORT void* malloc(size_t n_bytes);
LIBKRIGING_EXPORT void free(void* mem_ptr);
LIBKRIGING_EXPORT void set_allocation_functions(void* (*allocator)(size_t), void (*deallocator)(void*));
LIBKRIGING_EXPORT inline void unset_allocation_functions();
}  // namespace lkalloc

//#define ARMA_ALIEN_MEM_ALLOC_FUNCTION lkalloc::malloc
//#define ARMA_ALIEN_MEM_FREE_FUNCTION lkalloc::free
#ifndef LIBKRIGING_ARMA_ALIEN_MEM_FUNCTIONS_SET
#define LIBKRIGING_ARMA_ALIEN_MEM_FUNCTIONS_SET
#define CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET  // to cheat on carma
#endif

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LKALLOC_HPP
