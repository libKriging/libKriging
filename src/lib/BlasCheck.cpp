// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include <algorithm>
#include <cctype>
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

// BLAS identification (#429): OpenBLAS / MKL / BLIS entry points, when loaded.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#else
extern "C" {
char* openblas_get_config(void) __attribute__((weak));
char* openblas_get_corename(void) __attribute__((weak));
void mkl_get_version_string(char* buf, int len) __attribute__((weak));
char* bli_info_get_version_str(void) __attribute__((weak));
}
#endif

namespace {

using config_fn = char* (*)();
using mkl_version_fn = void (*)(char*, int);

template <typename F>
F blas_symbol(const char* name) {
#if defined(_WIN32)
  for (const char* dll : {"libopenblas.dll", "openblas.dll", "mkl_rt.2.dll", "mkl_rt.dll", "libblis.dll"}) {
    if (HMODULE h = GetModuleHandleA(dll))
      if (FARPROC p = GetProcAddress(h, name))
        return reinterpret_cast<F>(p);
  }
  return nullptr;
#elif defined(__APPLE__)
  return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, name));
#else
  const std::string n = name;
  if (n == "openblas_get_config")
    return reinterpret_cast<F>(&openblas_get_config);
  if (n == "openblas_get_corename")
    return reinterpret_cast<F>(&openblas_get_corename);
  if (n == "mkl_get_version_string")
    return reinterpret_cast<F>(&mkl_get_version_string);
  if (n == "bli_info_get_version_str")
    return reinterpret_cast<F>(&bli_info_get_version_str);
  return nullptr;
#endif
}

std::string blas_library() {
  if (auto config = blas_symbol<config_fn>("openblas_get_config")) {
    std::string s = config();
    if (auto core = blas_symbol<config_fn>("openblas_get_corename"))
      s += " | core " + std::string(core());
    return s;
  }
  if (auto version = blas_symbol<mkl_version_fn>("mkl_get_version_string")) {
    char buf[256] = {0};
    version(buf, sizeof(buf) - 1);
    std::string s = buf;
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
  }
  if (auto version = blas_symbol<config_fn>("bli_info_get_version_str"))
    return "BLIS " + std::string(version());
  return "unknown";
}

bool is_mkl_2020(const std::string& library) {
  return library.find("Math Kernel Library Version 2020") != std::string::npos;
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
  if (is_mkl_2020(library))
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

// One pass over the routines libKriging relies on; sizes above the
// small-matrix paths of the BLAS (the #429 DGEMM failed from 65^3 on).
Results run_checks(std::uint64_t seed) {
  Results res;
  std::mt19937_64 g(seed);
  const arma::uword n = 150;
  auto guarded = [&res](const std::string& name, const std::function<double()>& f) {
    double r;
    try {
      r = f();
    } catch (const std::exception&) {
      r = std::numeric_limits<double>::infinity();
    }
    keep_worst(res, name, r);
  };
  const arma::mat R = spd_mat(g, n);
  const arma::mat B = random_mat(g, n, 7);
  arma::mat L;
  try {
    L = LinearAlgebra::safe_chol_lower(R);
  } catch (const std::exception&) {
  }

  guarded("matrix product (dgemm)", [&] {
    const arma::mat a = random_mat(g, 120, 90), b = random_mat(g, 90, 130);
    const arma::mat c = a * b;
    return ratio(a, b, c, 1);
  });
  guarded("crossprod (dgemm / dsyrk)", [&] {
    const arma::mat a = random_mat(g, 140, 70);
    const arma::mat c = LinearAlgebra::crossprod(a);
    const arma::mat at = a.t();
    return ratio(at, a, c, 1);
  });
  guarded("Cholesky (dpotrf)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat lt = L.t();
    return ratio(L, lt, R, 4);
  });
  guarded("triangular solve (dtrsm / dtrtrs)", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat x = LinearAlgebra::solve_lower(L, B);
    const arma::mat lt = L.t();
    const arma::mat y = LinearAlgebra::solve_upper(lt, B);
    return std::max(ratio(L, x, B, 16), ratio(lt, y, B, 16));
  });
  guarded("general solve (dgesv)", [&] {
    arma::mat a = R + 0.1 * random_mat(g, n, n);
    const arma::mat x = LinearAlgebra::solve(a, B);
    return ratio(a, x, B, 16);
  });
  guarded("inverse from Cholesky", [&] {
    if (L.n_rows != n)
      return std::numeric_limits<double>::infinity();
    const arma::mat inv = LinearAlgebra::inv_sympd(L);
    return ratio(R, inv, arma::eye(n, n), 64);
  });
  guarded("rcond of Cholesky factor (*trcon / *gecon)", [&] {
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
  guarded("QR (dgeqrf / dorgqr)", [&] {
    const arma::mat a = random_mat(g, n, 40);
    arma::mat q, r;
    LinearAlgebra::qr_econ(q, r, a);
    const arma::mat qt = q.t();
    return std::max(ratio(q, r, a, 64), ratio(qt, q, arma::eye(q.n_cols, q.n_cols), 64));
  });
  guarded("symmetric eigen (dsyevd)", [&] {
    arma::vec w;
    arma::mat v;
    if (!arma::eig_sym(w, v, R))
      return std::numeric_limits<double>::infinity();
    arma::mat vw = v;
    vw.each_row() %= w.t();
    const arma::mat vt = v.t();
    return std::max(ratio(R, v, vw, 256), ratio(vt, v, arma::eye(n, n), 256));
  });
  return res;
}

}  // namespace

LIBKRIGING_EXPORT LinearAlgebra::BlasCheck LinearAlgebra::check_blas(int n_threads) {
  const auto t0 = std::chrono::steady_clock::now();
  if (n_threads <= 0)
    n_threads = std::max(2, static_cast<int>(std::thread::hardware_concurrency()));

  BlasCheck out;
  out.library = blas_library();

  // MKL 2020 with its default (Intel) threading layer next to the GNU OpenMP
  // runtime of libKriging gives NaN or crashes in dgetrf (#429): not run
  const char* layer = std::getenv("MKL_THREADING_LAYER");
  std::string lay = layer ? layer : "";
  std::transform(lay.begin(), lay.end(), lay.begin(), [](unsigned char c) { return std::toupper(c); });
  const bool skip = is_mkl_2020(out.library) && lay != "SEQUENTIAL" && lay != "GNU";

  const Results single = skip ? Results{} : run_checks(1);

  // the same checks from concurrent threads, as the multistart fits do
  std::vector<Results> per_thread(skip ? 0 : n_threads);
  if (!skip) {
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t)
      threads.emplace_back([t, &per_thread] {
        for (int rep = 0; rep < 2; ++rep)
          for (const auto& kv : run_checks(1000 + 100 * t + rep))
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
  std::string rep = "BLAS: " + out.library + "\n";
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
    std::snprintf(line, sizeof line, "%d concurrent threads:\n", n_threads);
    section(line, concurrent);
  }

  const std::string advice = known_issues(out.library);
  if (!advice.empty())
    rep += "known issue: " + advice + "\n";
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::snprintf(line, sizeof line, "%s in %.2f s\n", out.ok ? "all checks passed" : "SOME CHECKS FAILED", secs);
  rep += line;
  out.report = rep;
  return out;
}
