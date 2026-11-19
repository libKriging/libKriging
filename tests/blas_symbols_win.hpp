#ifndef LIBKRIGING_TESTS_BLAS_SYMBOLS_WIN_HPP
#define LIBKRIGING_TESTS_BLAS_SYMBOLS_WIN_HPP

// Windows counterpart of the weak BLAS symbols (openblas_get_num_threads,
// MKL_Get_Max_Threads, ...) the tests and benchmarks declare on ELF: PE / COFF
// has no weak symbols, so they are looked up at run time in the modules loaded
// in the process, whatever the DLL is called (openblas.dll from conda-forge,
// libopenblas.dll, mkl_rt.2.dll, ...). Include it last: it includes
// <windows.h>. Used by tests/BlasBackendTest.cpp and bench/bench-threads*.cpp,
// bench/bench-blas-threads.cpp (#473).

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// EnumProcessModules is K32EnumProcessModules of kernel32 (PSAPI_VERSION 2,
// Windows 7 and later): no psapi.lib to link
#include <psapi.h>

#include <string>
#include <vector>

namespace lk_blas_win {

/// First module of the process (in load order) exporting `name`, nullptr if none.
inline HMODULE module_exporting(const char* name) {
  std::vector<HMODULE> mods(256);
  DWORD needed = 0;
  for (;;) {
    const DWORD bytes = static_cast<DWORD>(mods.size() * sizeof(HMODULE));
    if (!EnumProcessModules(GetCurrentProcess(), mods.data(), bytes, &needed))
      return nullptr;
    if (needed <= bytes)
      break;
    mods.resize(needed / sizeof(HMODULE));
  }
  for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i)
    if (GetProcAddress(mods[i], name) != nullptr)
      return mods[i];
  return nullptr;
}

/// Address of the BLAS function `name` as a pointer of type F, nullptr if no
/// loaded module exports it.
template <typename F>
F proc(const char* name) {
  HMODULE m = module_exporting(name);
  return m ? reinterpret_cast<F>(GetProcAddress(m, name)) : nullptr;
}

/// Path of the module exporting `name`, "" if none.
inline std::string module_path(const char* name) {
  HMODULE m = module_exporting(name);
  if (m == nullptr)
    return "";
  char path[MAX_PATH] = {0};
  const DWORD len = GetModuleFileNameA(m, path, MAX_PATH);
  return std::string(path, len);
}

}  // namespace lk_blas_win

#endif  // _WIN32

#endif  // LIBKRIGING_TESTS_BLAS_SYMBOLS_WIN_HPP
