#include <iostream>
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

int main() {
  arma::arma_rng::set_seed(1);
  int n = 50, k = 5, m = 3;
  arma::mat U(n, k, arma::fill::randn);
  arma::vec D(n, arma::fill::randu);
  D += 0.5;  // keep positive, away from 0
  arma::mat B(n, m, arma::fill::randn);

  arma::mat ref = LinearAlgebra::woodbury_solve(U, D, B);

  LinearAlgebra::WoodburyFactorization wf(U, D);
  arma::mat got = wf.solve(B);

  std::cout << "max abs diff = " << arma::abs(ref - got).max() << std::endl;
  std::cout << "ref col0: " << ref.col(0).t();
  std::cout << "got col0: " << got.col(0).t();
  return 0;
}
