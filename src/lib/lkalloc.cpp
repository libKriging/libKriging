#include "libKriging/utils/lkalloc.hpp"

// #define LIBKRIGING_DEBUG_ALLOC

#ifdef LIBKRIGING_DEBUG_ALLOC
#include <iostream>
#include <unordered_set>
#endif

#include <cassert>
#include <cstdlib>

#if WIN32
#define _AMD64_
#include <libloaderapi.h>
#include <string>

HMODULE g_Handle;

EXTERN_C BOOL WINAPI DllMain(_In_ HINSTANCE hinstDLL, _In_ DWORD fdwReason, _In_opt_ LPVOID lpvReserved) {
  UNREFERENCED_PARAMETER(lpvReserved);

  if (fdwReason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(hinstDLL);
    g_Handle = hinstDLL;
  }

  return TRUE;
}

std::string dllName() {
  TCHAR dllName[MAX_PATH + 1];
  GetModuleFileName(g_Handle, dllName, MAX_PATH);
  return dllName;
}
#else
#include <string>
std::string dllName() {
  return {"<undefined>"};
}
#endif

namespace lkalloc {

void* (*custom_malloc)(size_t) = nullptr;
void (*custom_free)(void*) = nullptr;

#ifdef LIBKRIGING_DEBUG_ALLOC
std::unordered_set<void*> seens;
#endif

LIBKRIGING_EXPORT
void* malloc(size_t n_bytes) {
#ifdef LIBKRIGING_DEBUG_ALLOC
  static int count = 0;
  ++count;
  // std::cout << "Using lkalloc allocator " /* << custom_malloc */ << " (#" << count << ") in " << dllName() << "\n";
#endif
  void* mem_ptr = nullptr;
  if (custom_malloc) {
    mem_ptr = (*custom_malloc)(n_bytes);
  } else {
#ifdef _MSC_VER
    const size_t alignment = (n_bytes >= size_t(1024)) ? size_t(32) : size_t(16);
    mem_ptr = _aligned_malloc(n_bytes, alignment);
#else
    mem_ptr = ::malloc(n_bytes);
#endif
  }
#ifdef LIBKRIGING_DEBUG_ALLOC
  seens.insert(mem_ptr);
#endif
  return mem_ptr;
}

LIBKRIGING_EXPORT
void free(void* mem_ptr) {
#ifdef LIBKRIGING_DEBUG_ALLOC
  static int count = 0;
  ++count;
  // std::cout << "Using lkalloc deallocator " /* << custom_free */ << " (#" << count << ") in " << dllName() << "\n";
  if (seens.find(mem_ptr) == seens.end()) {
    std::cout << "### (#" << count << ") lkalloc allocator has never seen " << mem_ptr << " ##" << std::endl;
    return;
  }
#endif
  if (custom_free) {
    (*custom_free)(mem_ptr);
  } else {
#ifdef _MSC_VER
    return _aligned_free(mem_ptr);
#else
    ::free(mem_ptr);
#endif
  }
#ifdef LIBKRIGING_DEBUG_ALLOC
  seens.erase(mem_ptr);
#endif
}

LIBKRIGING_EXPORT
void set_allocation_functions(void* (*allocator)(size_t), void (*deallocator)(void*)) {
  assert(custom_malloc == nullptr && custom_free == nullptr);
  assert(allocator != nullptr && deallocator != nullptr);
  custom_malloc = allocator;
  custom_free = deallocator;
}

LIBKRIGING_EXPORT
void unset_allocation_functions() {
  assert(custom_malloc != nullptr && custom_free != nullptr);
  custom_malloc = nullptr;
  custom_free = nullptr;
}

}  // namespace lkalloc
