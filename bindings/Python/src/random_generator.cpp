#include "random_generator.hpp"
#include <random>

py::array_t<double> generate_uniform_random_array(const arma::uword nrows, const arma::uword ncols) {
  std::mt19937 mersenne_engine{/* seed = */ 123};  // Unconstrained random generator
  std::uniform_real_distribution<double> dist{};   // With a distribution model
  arma::mat r(nrows, ncols);
  r.imbue([&]() { return dist(mersenne_engine); });
  return carma::mat_to_arr(r, true);
}

void set_random_seed(const arma::arma_rng::seed_type seed) {
  arma::arma_rng::set_seed(seed);
}
