#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
template <class F> double med_ms(int reps, F&& f) {
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) { auto t0 = std::chrono::steady_clock::now(); f(); t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
  std::nth_element(t.begin(), t.begin() + reps / 2, t.end()); return t[reps / 2];
}
int main() {
  const double a[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898}, th[3] = {0.3, 0.5, 0.7};
  const arma::vec thv = {0.3, 0.5, 0.7};
  printf("n, potri+symmatl (ms), inv_sympd with guard (ms), guard overhead (ms, %%)\n");
  for (arma::uword n : {500u, 1000u, 1500u, 2000u, 3000u}) {
    arma::mat X(n, 3);
    for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < 3; ++j) X(i, j) = std::fmod(0.5 + (i + 1) * a[j], 1.0);
    arma::mat R(n, n);
    for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < n; ++j) R(i, j) = std::exp(-0.5 * arma::accu(arma::square((X.row(i) - X.row(j)).t() / thv)));
    R.diag() += 1e-9; const arma::mat L = arma::chol(R, "lower");
    int reps = n <= 1000 ? 25 : 9;
    arma::mat sink;
    double raw = med_ms(reps, [&] { arma::mat P = L; char u = 'L'; arma::blas_int nn = n, info = 0; arma::lapack::potri(&u, &nn, P.memptr(), &nn, &info); sink = arma::symmatl(P); });
    double lib = med_ms(reps, [&] { sink = LinearAlgebra::inv_sympd(L); });
    printf("%u, %.2f, %.2f, %.2f (%.1f%%)\n", (unsigned)n, raw, lib, lib - raw, 100.0 * (lib - raw) / raw);
  }
}
