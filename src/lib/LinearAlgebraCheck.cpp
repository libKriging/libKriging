// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

// Backend identification (#429): the module providing dgemm_ (whatever the BLAS:
// OpenBLAS, MKL, BLIS, FlexiBLAS, Accelerate, reference), then the vendor's
// own version entry points when it has some.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#if defined(_WIN32)
// candidate BLAS DLLs, vendor libraries first (conda's libblas.dll forwards to one of them)
const char* const kWinBlasDlls[] = {"libopenblas.dll",
                                    "openblas.dll",
                                    "mkl_rt.2.dll",
                                    "mkl_rt.dll",
                                    "libblis.dll",
                                    "libflexiblas.dll",
                                    "Rblas.dll",
                                    "libblas.dll",
                                    "blas.dll"};
#endif

// address of an exported symbol of the process (nullptr if absent), and the
// file of the module exporting it ("" if unknown)
template <typename F>
F blas_symbol(const char* name, std::string* module = nullptr) {
#if defined(_WIN32)
  for (const char* dll : kWinBlasDlls) {
    HMODULE h = GetModuleHandleA(dll);
    if (!h)
      continue;
    FARPROC p = GetProcAddress(h, name);
    if (!p)
      continue;
    if (module) {
      char path[MAX_PATH] = {0};
      GetModuleFileNameA(h, path, MAX_PATH);
      *module = path;
    }
    return reinterpret_cast<F>(reinterpret_cast<void (*)()>(p));
  }
  return nullptr;
#else
  void* p = dlsym(RTLD_DEFAULT, name);
  if (p && module) {
    Dl_info info;
    if (dladdr(p, &info) && info.dli_fname) {
      // the file behind the symlinks (e.g. libblas.so.3 -> openblas-pthread/...)
      char* real = realpath(info.dli_fname, nullptr);
      *module = real ? real : info.dli_fname;
      std::free(real);
    }
  }
  return reinterpret_cast<F>(p);
#endif
}

using config_fn = char* (*)();
using mkl_version_fn = void (*)(char*, int);
using flexiblas_backend_fn = int (*)(char*, size_t);

std::string detect_backend() {
  std::string module;
  (void)blas_symbol<void (*)()>("dgemm_", &module);
  std::string vendor;
  if (auto backend = blas_symbol<flexiblas_backend_fn>("flexiblas_current_backend")) {
    char buf[256] = {0};
    backend(buf, sizeof(buf) - 1);
    vendor = "FlexiBLAS, backend " + std::string(buf);
  } else if (auto config = blas_symbol<config_fn>("openblas_get_config")) {
    vendor = config();
    if (auto core = blas_symbol<config_fn>("openblas_get_corename"))
      vendor += " | core " + std::string(core());
  } else if (auto version = blas_symbol<mkl_version_fn>("mkl_get_version_string")) {
    char buf[256] = {0};
    version(buf, sizeof(buf) - 1);
    vendor = buf;
    vendor.erase(vendor.find_last_not_of(' ') + 1);
    const char* layer = std::getenv("MKL_THREADING_LAYER");
    vendor += " | threading layer " + std::string(layer ? layer : "INTEL (default)");
  } else if (auto version = blas_symbol<config_fn>("bli_info_get_version_str")) {
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
      vendor = "reference BLAS (Netlib) or unknown";
    else
      vendor = "unknown vendor";
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
  // LK_LINEARALGEBRA_CHECK_TRACE: one stderr line per check, to locate a BLAS crash
  static const bool trace = std::getenv("LK_LINEARALGEBRA_CHECK_TRACE") != nullptr;
  auto guarded = [&res, seed](const std::string& name, const std::function<double()>& f) {
    if (trace) {
      std::fprintf(stderr, "LinearAlgebra::check[%llu] %s\n", static_cast<unsigned long long>(seed), name.c_str());
      std::fflush(stderr);
    }
    const auto t0 = std::chrono::steady_clock::now();
    double r;
    try {
      r = f();
    } catch (const std::exception&) {
      r = std::numeric_limits<double>::infinity();
    }
    if (trace) {
      std::fprintf(stderr,
                   "LinearAlgebra::check[%llu] %s done in %.3f s\n",
                   static_cast<unsigned long long>(seed),
                   name.c_str(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stderr);
    }
    keep_worst(res, name, r);
  };
  const arma::mat R = spd_mat(g, n);
  const arma::mat B = random_mat(g, n, 7);
  arma::mat L;
  try {
    if (trace) {
      std::fprintf(stderr, "LinearAlgebra::check[%llu] safe_chol_lower\n", static_cast<unsigned long long>(seed));
      std::fflush(stderr);
    }
    L = LinearAlgebra::safe_chol_lower(R);
  } catch (const std::exception&) {
  }

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

  const Results single = skip ? Results{} : run_checks(1, false);
  const auto t2 = std::chrono::steady_clock::now();

  // the same checks from concurrent threads, as the multistart fits do
  std::vector<Results> per_thread(skip ? 0 : n_threads);
  if (!skip) {
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t)
      threads.emplace_back([t, &per_thread] {
        for (int rep = 0; rep < 2; ++rep)
          for (const auto& kv : run_checks(1000 + 100 * t + rep, true))
            keep_worst(per_thread[t], kv.first, kv.second);
      });
    for (auto& th : threads)
      th.join();
  }
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
