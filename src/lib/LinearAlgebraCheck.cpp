// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include "ThreadBudget.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

// Backend identification (#429): the module libKriging's dgemm is bound to
// (whatever the BLAS: OpenBLAS, MKL, BLIS, FlexiBLAS, Accelerate, reference),
// then the vendor's own version entry points in that module when it has some.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(__linux__)
#include <link.h>
#endif
#endif

namespace {

// The module libKriging's dgemm is bound to: Armadillo's dgemm entry, not the
// first dgemm_ of the process (on macOS, two-level namespace, Octave may load
// OpenBLAS next to the Accelerate the library is linked to).
struct BlasModule {
  std::string path;  // "" if unknown
#if defined(_WIN32)
  HMODULE handle = nullptr;
#else
  void* handle = nullptr;
#endif
  BlasModule() = default;
  BlasModule(const BlasModule&) = delete;
  BlasModule& operator=(const BlasModule&) = delete;
  ~BlasModule() {
#if !defined(_WIN32)
    if (handle)
      dlclose(handle);
#endif
  }
};

void bound_module(BlasModule& m) {
#if defined(ARMA_USE_BLAS)
  const void* p = reinterpret_cast<const void*>(&arma::arma_fortran(arma_dgemm));
#if defined(_WIN32)
#if defined(_WIN64)
  // without dllimport, &f is a stub of the calling module: follow the
  // incremental-link "jmp rel32" and the import "jmp [rip + disp32]"
  for (int hops = 0; hops < 4 && p; ++hops) {
    const auto* b = static_cast<const unsigned char*>(p);
    std::int32_t disp;
    if (b[0] == 0xE9) {
      std::memcpy(&disp, b + 1, 4);
      p = b + 5 + disp;
    } else if (b[0] == 0xFF && b[1] == 0x25) {
      std::memcpy(&disp, b + 2, 4);
      p = *reinterpret_cast<void* const*>(b + 6 + disp);
    } else {
      break;
    }
  }
#endif
  HMODULE h = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(p),
                          &h)
      || !GetProcAddress(h, "dgemm_")) {
    // not the BLAS itself (Armadillo's wrapper DLL, a static link): the
    // vendor DLL loaded in the process (conda's libblas.dll forwards to one)
    h = nullptr;
    for (const char* dll : {"libopenblas.dll",
                            "openblas.dll",
                            "mkl_rt.2.dll",
                            "mkl_rt.dll",
                            "libblis.dll",
                            "libflexiblas.dll",
                            "Rblas.dll",
                            "libblas.dll",
                            "blas.dll"})
      if ((h = GetModuleHandleA(dll)) && GetProcAddress(h, "dgemm_"))
        break;
      else
        h = nullptr;
  }
  if (h) {
    char path[MAX_PATH] = {0};
    GetModuleFileNameA(h, path, MAX_PATH);
    m.path = path;
    m.handle = h;
  }
#else
  for (int hops = 0; hops < 2; ++hops) {
    Dl_info info;
    if (!dladdr(p, &info) || !info.dli_fname)
      break;
    if (m.handle)
      dlclose(m.handle);
    // the file behind the symlinks (e.g. libblas.so.3 -> openblas-pthread/...)
    char* real = realpath(info.dli_fname, nullptr);
    m.path = real ? real : info.dli_fname;
    std::free(real);
    m.handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    // not the BLAS itself (Armadillo's wrapper library): the dgemm_ of its dependencies
    const void* q = m.handle ? dlsym(m.handle, "dgemm_") : nullptr;
    if (!q || q == p)
      break;
    p = q;
  }
#if defined(__linux__)
  // Armadillo's wrapper linked into a non-PIE executable (static libKriging,
  // e.g. the Octave builds): wrapper2_dgemm_ is in the executable, whose path
  // dlopen does not reopen; the first shared object of the load order
  // defining dgemm_ is the one the wrapper calls
  char* exe = realpath("/proc/self/exe", nullptr);
  const bool in_exe = exe && m.path == exe;
  std::free(exe);
  if (in_exe) {
    struct Found {
      std::string path;
    } found;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t, void* data) -> int {
          if (!info->dlpi_name || !info->dlpi_name[0])
            return 0;  // the executable
          void* h = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
          if (!h)
            return 0;
          void* q = dlsym(h, "dgemm_");
          Dl_info qi;
          const bool own = q && dladdr(q, &qi) && qi.dli_fname && std::string(qi.dli_fname) == info->dlpi_name;
          dlclose(h);
          if (!own)
            return 0;
          static_cast<Found*>(data)->path = info->dlpi_name;
          return 1;
        },
        &found);
    if (!found.path.empty()) {
      if (m.handle)
        dlclose(m.handle);
      char* real = realpath(found.path.c_str(), nullptr);
      m.path = real ? real : found.path;
      std::free(real);
      m.handle = dlopen(found.path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    }
  }
#endif
#endif
#else
  (void)m;
#endif
}

// address of a vendor entry point in that module (nullptr if absent)
template <typename F>
F blas_symbol(const BlasModule& m, const char* name) {
  if (!m.handle)
    return nullptr;
#if defined(_WIN32)
  return reinterpret_cast<F>(reinterpret_cast<void (*)()>(GetProcAddress(m.handle, name)));
#else
  return reinterpret_cast<F>(dlsym(m.handle, name));
#endif
}

using config_fn = char* (*)();
using mkl_version_fn = void (*)(char*, int);
using flexiblas_backend_fn = int (*)(char*, size_t);

std::string detect_backend() {
  BlasModule m;
  bound_module(m);
  const std::string& module = m.path;
  std::string vendor;
  if (auto backend = blas_symbol<flexiblas_backend_fn>(m, "flexiblas_current_backend")) {
    char buf[256] = {0};
    backend(buf, sizeof(buf) - 1);
    vendor = "FlexiBLAS, backend " + std::string(buf);
  } else if (auto config = blas_symbol<config_fn>(m, "openblas_get_config")) {
    vendor = config();
    if (auto core = blas_symbol<config_fn>(m, "openblas_get_corename"))
      vendor += " | core " + std::string(core());
  } else if (auto version = blas_symbol<mkl_version_fn>(m, "mkl_get_version_string")) {
    char buf[256] = {0};
    version(buf, sizeof(buf) - 1);
    vendor = buf;
    vendor.erase(vendor.find_last_not_of(' ') + 1);
    const char* layer = std::getenv("MKL_THREADING_LAYER");
    vendor += " | threading layer " + std::string(layer ? layer : "INTEL (default)");
  } else if (auto version = blas_symbol<config_fn>(m, "bli_info_get_version_str")) {
    vendor = "BLIS " + std::string(version());
  } else {
    // no version entry point: guess from the file name
    std::string f = module;
    std::transform(f.begin(), f.end(), f.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    auto has = [&f](const char* x) { return f.find(x) != std::string::npos; };
    if (has("accelerate") || has("veclib"))
      vendor = "Apple Accelerate";
    else if (has("blis"))
      vendor = "BLIS";
    else if (has("flexiblas"))
      vendor = "FlexiBLAS";
    else if (has("openblas"))
      vendor = "OpenBLAS";
    else if (has("mkl"))
      vendor = "MKL";
    else if (has("rblas"))
      vendor = "R reference BLAS";
    else if (has("/blas/") || has("libblas") || has("\\blas"))
      vendor = "unidentified BLAS, reference (Netlib) by its name";
    else
      vendor = "unidentified BLAS";
  }
  return module.empty() ? vendor : vendor + " (" + module + ")";
}

// MKL 2020 with its default Intel threading layer next to the GNU OpenMP
// runtime of libKriging gives NaN or crashes in dgetrf (#429)
bool is_mkl_2020_unsafe(const std::string& library) {
  return library.find("Math Kernel Library Version 2020") != std::string::npos
         && library.find("threading layer GNU") == std::string::npos
         && library.find("threading layer gnu") == std::string::npos
         && library.find("threading layer SEQUENTIAL") == std::string::npos
         && library.find("threading layer sequential") == std::string::npos;
}

// Configurations known to give wrong results (#429), with their workaround.
std::string known_issues(const std::string& library) {
  std::smatch m;
  std::string advice;
  if (std::regex_search(library, m, std::regex(R"(OpenBLAS 0\.3\.(19|20)\b.*\| core (\w+))"))) {
    const std::string version = m[1], core = m[2];
    if (core == "Cooperlake" || (version == "19" && core == "SkylakeX"))
      advice = "OpenBLAS 0.3." + version + " " + core
               + " kernels give wrong DGEMM / TRSM results (#429): set OPENBLAS_CORETYPE="
               + (version == "20" ? "SkylakeX" : "Haswell")
               + " in the environment before starting the process, or use OpenBLAS >= 0.3.21.";
  }
  if (is_mkl_2020_unsafe(library))
    advice
        = "MKL 2020 with its default Intel threading layer gives NaN results or crashes next to the "
          "GNU OpenMP runtime of libKriging (#429): set MKL_THREADING_LAYER=GNU (or SEQUENTIAL) before "
          "starting the process, or use a recent MKL.";
  return advice;
}

constexpr double kEps = std::numeric_limits<double>::epsilon();
// a check fails beyond kTol times its rounding bound; a wrong BLAS kernel
// gives relative errors of order 1, i.e. ratios of order 1e13 and more
constexpr double kTol = 100;

// Matrices from a local generator: the Armadillo RNG state of the caller is
// left untouched, and nothing goes through the BLAS under test.
arma::mat random_mat(std::mt19937_64& g, arma::uword n, arma::uword m) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  arma::mat a(n, m);
  for (arma::uword k = 0; k < a.n_elem; ++k)
    a[k] = u(g);
  return a;
}

// G G^T / m + I, by hand
arma::mat spd_mat(std::mt19937_64& g, arma::uword n) {
  arma::mat x = random_mat(g, n, n);
  arma::mat a(n, n);
  for (arma::uword j = 0; j < n; ++j)
    for (arma::uword i = 0; i < n; ++i) {
      long double s = 0;
      for (arma::uword l = 0; l < n; ++l)
        s += static_cast<long double>(x.at(i, l)) * x.at(j, l);
      a.at(i, j) = static_cast<double>(s / n) + (i == j ? 1.0 : 0.0);
    }
  return a;
}

// max_ij |(A X)_ij - T_ij| / (((|A| |X|)_ij + |T_ij|) (k + 2) eps factor), the
// product A X computed in long double; NaN-propagating
double ratio(const arma::mat& a, const arma::mat& x, const arma::mat& t, double factor) {
  if (a.n_cols != x.n_rows || a.n_rows != t.n_rows || x.n_cols != t.n_cols)
    return std::numeric_limits<double>::infinity();
  const double k = static_cast<double>(a.n_cols);
  double worst = 0;
  for (arma::uword j = 0; j < t.n_cols; ++j)
    for (arma::uword i = 0; i < t.n_rows; ++i) {
      long double s = 0, sa = 0;
      for (arma::uword l = 0; l < a.n_cols; ++l) {
        const long double p = static_cast<long double>(a.at(i, l)) * x.at(l, j);
        s += p;
        sa += std::fabs(p);
      }
      const long double bound = (sa + std::fabs(t.at(i, j))) * (k + 2) * kEps * factor + 1e-300L;
      const double r = static_cast<double>(std::fabs(s - t.at(i, j)) / bound);
      if (!(r <= worst))
        worst = r;
    }
  return worst;
}

using Results = std::map<std::string, double>;

// LK_LINEARALGEBRA_CHECK_TRACE: stderr lines per check and per worker, to locate a BLAS crash or stall
bool trace_enabled() {
  static const bool trace = std::getenv("LK_LINEARALGEBRA_CHECK_TRACE") != nullptr;
  return trace;
}

// one trace line, stamped with the time since the first one: the stamps of a
// captured output (e.g. ctest) are those of its flushes, not of the events
void trace_line(const char* fmt, ...) {
  if (!trace_enabled())
    return;
  static const auto start = std::chrono::steady_clock::now();
  char msg[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(msg, sizeof msg, fmt, args);
  va_end(args);
  std::fprintf(stderr,
               "LinearAlgebra::check +%.3f s %s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
               msg);
  std::fflush(stderr);
}

void keep_worst(Results& into, const std::string& name, double r) {
  auto it = into.find(name);
  if (it == into.end() || !(r <= it->second))
    into[name] = r;
}

// One pass over the LinearAlgebra routines libKriging relies on; sizes above the
// small-matrix paths of the BLAS (the #429 DGEMM failed from 65^3 on).
// concurrent: only the routines libKriging calls from its thread pool (the
// symmetric eigen decomposition is used by simulate / update_simulate only,
// and crashed concurrently with MKL on Windows, MSVC OpenMP next to Intel's).
Results run_checks(std::uint64_t seed, bool concurrent) {
  Results res;
  std::mt19937_64 g(seed);
  const arma::uword n = 150;
  const unsigned long long sd = seed;
  trace_line("[%llu] start", sd);
  auto guarded = [&res, sd](const std::string& name, const std::function<double()>& f) {
    trace_line("[%llu] %s", sd, name.c_str());
    double r;
    try {
      r = f();
    } catch (const std::exception&) {
      r = std::numeric_limits<double>::infinity();
    }
    trace_line("[%llu] %s done", sd, name.c_str());
    keep_worst(res, name, r);
  };
  const arma::mat R = spd_mat(g, n);
  const arma::mat B = random_mat(g, n, 7);
  arma::mat L;
  trace_line("[%llu] matrices drawn; safe_chol_lower", sd);
  try {
    L = LinearAlgebra::safe_chol_lower(R);
  } catch (const std::exception&) {
  }
  trace_line("[%llu] safe_chol_lower done", sd);

  // each check: one public LinearAlgebra call, against an oracle computed
  // here in long double without the BLAS
  guarded("crossprod (dgemm / dsyrk)", [&] {
    const arma::mat a = random_mat(g, 140, 70);
    const arma::mat c = LinearAlgebra::crossprod(a);
    const arma::mat at = a.t();
    return ratio(at, a, c, 1);
  });
  guarded("diagABA (dgemm)", [&] {
    // diagABA(A, B)_i = a_i^T B a_i for a symmetric B, a_i the i-th row of A
    const arma::mat a = random_mat(g, n, 130);
    arma::mat b = random_mat(g, 130, 130);
    b = 0.5 * (b + b.t());
    const arma::vec d = LinearAlgebra::diagABA(a, b);
    double worst = 0;
    for (arma::uword i = 0; i < a.n_rows; ++i) {
      long double s = 0, sa = 0;
      for (arma::uword k = 0; k < b.n_rows; ++k)
        for (arma::uword j = 0; j < b.n_cols; ++j) {
          const long double p = static_cast<long double>(a.at(i, k)) * b.at(k, j) * a.at(i, j);
          s += p;
          sa += std::fabs(p);
        }
      const double r
          = static_cast<double>(std::fabs(s - d[i]) / ((sa + std::fabs(d[i])) * (b.n_rows + 2) * kEps * 4 + 1e-300L));
      if (!(r <= worst))
        worst = r;
    }
    return worst;
  });
  guarded("safe_chol_lower (dpotrf)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat lt = L.t();
    return ratio(L, lt, R, 4);
  });
  guarded("chol_upper (dpotrf)", [&] {
    const arma::mat u = LinearAlgebra::chol_upper(R);
    const arma::mat ut = u.t();
    return ratio(ut, u, R, 4);
  });
  guarded("solve_lower / solve_upper (dtrsm)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat x = LinearAlgebra::solve_lower(L, B);
    const arma::mat lt = L.t();
    const arma::mat y = LinearAlgebra::solve_upper(lt, B);
    return std::max(ratio(L, x, B, 16), ratio(lt, y, B, 16));
  });
  guarded("rsolve_upper (dtrsm)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat u = L.t(), bt = B.t();
    const arma::mat x = LinearAlgebra::rsolve_upper(u, bt);  // X U = B^T
    return ratio(x, u, bt, 16);
  });
  guarded("solve (dgesv)", [&] {
    arma::mat a = R + 0.1 * random_mat(g, n, n);
    const arma::mat x = LinearAlgebra::solve(a, B);
    return ratio(a, x, B, 16);
  });
  guarded("inv_sympd", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat inv = LinearAlgebra::inv_sympd(L);
    return ratio(R, inv, arma::eye(n, n), 64);
  });
  guarded("rcond_chol (*trcon / *gecon)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    // exact 1-norm condition of L, from its inverse computed by hand
    arma::mat inv(n, n, arma::fill::zeros);
    for (arma::uword j = 0; j < n; ++j)
      for (arma::uword i = j; i < n; ++i) {
        long double s = (i == j) ? 1.0L : 0.0L;
        for (arma::uword l = j; l < i; ++l)
          s -= static_cast<long double>(L.at(i, l)) * inv.at(l, j);
        inv.at(i, j) = static_cast<double>(s / L.at(i, i));
      }
    const double exact = 1.0 / (arma::norm(L, 1) * arma::norm(inv, 1));
    const double est = std::sqrt(LinearAlgebra::rcond_chol(L));  // rcond_chol squares it
    // LAPACK underestimates ||L^-1||_1: est >= exact, within a factor ~n
    const double q = est / exact;
    return (q > 1 - 1e-6 && q < 10.0 * n) ? 0.0 : std::numeric_limits<double>::infinity();
  });
  guarded("qr_econ (dgeqrf / dorgqr)", [&] {
    const arma::mat a = random_mat(g, n, 40);
    arma::mat q, r;
    LinearAlgebra::qr_econ(q, r, a);
    const arma::mat qt = q.t();
    return std::max(ratio(q, r, a, 64), ratio(qt, q, arma::eye(q.n_cols, q.n_cols), 64));
  });
  guarded("woodbury_solve (dgemm / dgesv)", [&] {
    // (diag(D) + U U^T) X = B, the matrix built by hand
    const arma::mat u = random_mat(g, n, 20);
    const arma::mat r = random_mat(g, n, 1);
    arma::vec dv(n);
    for (arma::uword i = 0; i < n; ++i)
      dv[i] = 1.0 + std::fabs(r.at(i, 0));
    const arma::mat x = LinearAlgebra::woodbury_solve(u, dv, B);
    arma::mat m(n, n);
    for (arma::uword j = 0; j < n; ++j)
      for (arma::uword i = 0; i < n; ++i) {
        long double s = (i == j) ? dv[i] : 0.0L;
        for (arma::uword l = 0; l < u.n_cols; ++l)
          s += static_cast<long double>(u.at(i, l)) * u.at(j, l);
        m.at(i, j) = static_cast<double>(s);
      }
    return ratio(m, x, B, 64);
  });
  if (concurrent)
    return res;
  guarded("eig_sym (dsyevd)", [&] {
    arma::vec w;
    arma::mat v;
    if (!LinearAlgebra::eig_sym(w, v, R))
      return std::numeric_limits<double>::infinity();
    arma::mat vw = v;
    vw.each_row() %= w.t();
    const arma::mat vt = v.t();
    return std::max(ratio(R, v, vw, 256), ratio(vt, v, arma::eye(n, n), 256));
  });
  return res;
}

}  // namespace

LIBKRIGING_EXPORT LinearAlgebra::CheckResult LinearAlgebra::check(int n_threads) {
  const auto t0 = std::chrono::steady_clock::now();
  if (n_threads <= 0)
    n_threads = std::max(2, static_cast<int>(std::thread::hardware_concurrency()));

  CheckResult out;
  out.backend = detect_backend();
  const auto t1 = std::chrono::steady_clock::now();

  // not run where it would crash (MKL 2020, Intel threading layer)
  const bool skip = is_mkl_2020_unsafe(out.backend);

  // the thread budget of the public methods (OpenBLAS count included)
  trace_line("budget scope");
  libKriging::detail::BudgetScope budget;
  trace_line("budget scope set");

  const Results single = skip ? Results{} : run_checks(1, false);
  const auto t2 = std::chrono::steady_clock::now();

  // the same checks from concurrent threads, in the conditions of the task
  // pool of the multistart fits: each worker marked as such, with its share
  // of the budget for the BLAS (OpenBLAS count set once for all of them)
  std::vector<Results> per_thread(skip ? 0 : n_threads);
  if (!skip) {
    const int share = std::max(1, libKriging::detail::budget_total() / n_threads);
    trace_line("concurrent pass: %d workers, BLAS share %d, OpenBLAS count %d",
               n_threads,
               share,
               libKriging::detail::blas_threads_for(share));
    libKriging::detail::ScopedBlasThreads blas_share(libKriging::detail::blas_threads_for(share));
    trace_line("OpenBLAS count set");
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t)
      threads.emplace_back([t, share, &per_thread] {
        trace_line("worker %d started", t);
        libKriging::detail::ScopedShare worker_share(share);
        for (int rep = 0; rep < 2; ++rep)
          for (const auto& kv : run_checks(1000 + 100 * t + rep, true))
            keep_worst(per_thread[t], kv.first, kv.second);
        trace_line("worker %d done", t);
      });
    trace_line("workers created");
    for (auto& th : threads)
      th.join();
    trace_line("workers joined; OpenBLAS count restored next");
  }
  trace_line("concurrent pass closed");
  Results concurrent;
  for (const auto& r : per_thread)
    for (const auto& kv : r)
      keep_worst(concurrent, kv.first, kv.second);

  out.ok = true;
  std::string rep = "backend: " + out.backend + "\n";
  char line[256];
  auto section = [&](const char* title, const Results& r) {
    rep += title;
    for (const auto& kv : r) {
      const bool ok = kv.second <= kTol;
      out.ok = out.ok && ok;
      std::snprintf(line,
                    sizeof line,
                    "  %-44s %s  (error / rounding bound: %.3g)\n",
                    kv.first.c_str(),
                    ok ? "ok    " : "FAILED",
                    kv.second);
      rep += line;
    }
  };
  if (skip) {
    out.ok = false;
    rep += "checks not run: MKL 2020 with its Intel threading layer (see below)\n";
  } else {
    section("single thread:\n", single);
    std::snprintf(line, sizeof line, "%d concurrent threads (no eigen decomposition):\n", n_threads);
    section(line, concurrent);
  }

  const std::string advice = known_issues(out.backend);
  if (!advice.empty())
    rep += "known issue: " + advice + "\n";
  const auto t3 = std::chrono::steady_clock::now();
  auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::snprintf(line,
                sizeof line,
                "%s in %.2f s (backend detection %.2f s, single thread %.2f s, concurrent %.2f s)\n",
                out.ok ? "all checks passed" : "SOME CHECKS FAILED",
                secs(t0, t3),
                secs(t0, t1),
                secs(t1, t2),
                secs(t2, t3));
  rep += line;
  out.report = rep;
  return out;
}
