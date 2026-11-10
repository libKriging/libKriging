#include "libKriging/utils/lk_armadillo.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
template <class F> double med_ms(int reps, F&& f) {
  std::vector<double> t; for (int r = 0; r < reps; ++r) { auto t0 = std::chrono::steady_clock::now(); f(); t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
  std::nth_element(t.begin(), t.begin() + reps / 2, t.end()); return t[reps / 2];
}
int main() {
  const arma::vec thv = {0.3, 0.5, 0.7}; const double a[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898};
  printf("n | L.t() | solve x2 (current) | solve x2 fast | cblas_dtrsv x2 | Rinv*v   (ms, median)  | max |w_current - w_dtrsv|/|w|\n");
  for (arma::uword n : {500u, 1000u, 2000u, 3000u}) {
    arma::mat X(n, 3); for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < 3; ++j) X(i, j) = std::fmod(0.5 + (i + 1) * a[j], 1.0);
    arma::mat R(n, n); for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < n; ++j) R(i, j) = std::exp(-0.5 * arma::accu(arma::square((X.row(i) - X.row(j)).t() / thv)));
    R.diag() += 1e-9; const arma::mat L = arma::chol(R, "lower"); const arma::mat Rinv = arma::inv_sympd(R);
    arma::vec v(n); for (arma::uword i = 0; i < n; ++i) v(i) = std::fmod(0.5 + (i + 1) * 0.6180339887498949, 1.0) - 0.5;
    int reps = 15; arma::vec w, w2, w3; arma::mat Lt;
    double t_t = med_ms(reps, [&] { Lt = L.t(); });
    double t_cur = med_ms(reps, [&] { w = arma::solve(arma::trimatu(L.t()), arma::solve(arma::trimatl(L), v)); });
    double t_fast = med_ms(reps, [&] { w2 = arma::solve(arma::trimatu(L.t()), arma::solve(arma::trimatl(L), v, arma::solve_opts::fast), arma::solve_opts::fast); });
    double t_trsv = med_ms(reps, [&] { w3 = v; cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L.memptr(), n, w3.memptr(), 1); cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, L.memptr(), n, w3.memptr(), 1); });
    arma::vec y; double t_mv = med_ms(reps, [&] { y = Rinv * v; });
    printf("%u | %.2f | %.2f | %.2f | %.2f | %.2f | %.2g\n", (unsigned)n, t_t, t_cur, t_fast, t_trsv, t_mv, arma::norm(w - w3, "inf") / arma::norm(w, "inf"));
  }
}
