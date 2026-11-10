#include "libKriging/utils/lk_armadillo.hpp"
#include <cstdio>
// same check as the new inv_sympd guard (two trtrs on a fixed vector), applied to a correct and to corrupted inverses
static double gap(const arma::mat& L, const arma::mat& Rinv) {
  const arma::blas_int n = L.n_rows; arma::vec v(n);
  for (arma::uword i = 0; i < v.n_elem; ++i) v.at(i) = std::fmod(0.5 + (i + 1) * 0.6180339887498949, 1.0) - 0.5;
  arma::vec w = v; char lo = 'L', no = 'N', tr = 'T', nu = 'N'; arma::blas_int one = 1, i1 = 0, i2 = 0, nn = n;
  arma::lapack::trtrs(&lo, &no, &nu, &nn, &one, const_cast<double*>(L.memptr()), &nn, w.memptr(), &nn, &i1);
  arma::lapack::trtrs(&lo, &tr, &nu, &nn, &one, const_cast<double*>(L.memptr()), &nn, w.memptr(), &nn, &i2);
  return arma::norm(Rinv * v - w, "inf") / arma::norm(w, "inf");
}
int main() {
  const arma::uword n = 750; const arma::vec th = {0.3, 0.5, 0.7}; const double a[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898};
  arma::mat X(n, 3); for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < 3; ++j) X(i, j) = std::fmod(0.5 + (i + 1) * a[j], 1.0);
  arma::mat R(n, n); for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < n; ++j) R(i, j) = std::exp(-0.5 * arma::accu(arma::square((X.row(i) - X.row(j)).t() / th)));
  R.diag() += 1e-9; const arma::mat L = arma::chol(R, "lower");
  arma::mat good = arma::solve(arma::trimatu(L.t()), arma::solve(arma::trimatl(L), arma::eye(n, n)));
  printf("correct inverse:                       gap %.3g (threshold 1e-8)\n", gap(L, good));
  arma::mat bad = good; bad.submat(0, 0, n / 2 - 1, n / 2 - 1).zeros();
  printf("top-left half block left unprocessed:  gap %.3g -> %s\n", gap(L, bad), gap(L, bad) > 1e-8 ? "DETECTED" : "missed");
  arma::mat bad2 = good; bad2(100, 50) *= 1.0 + 1e-6; bad2(50, 100) = bad2(100, 50);
  printf("one entry off by 1e-6 relative:        gap %.3g -> %s\n", gap(L, bad2), gap(L, bad2) > 1e-8 ? "DETECTED" : "missed");
}
