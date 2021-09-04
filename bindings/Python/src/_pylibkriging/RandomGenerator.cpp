#include "RandomGenerator.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <random>

RandomGenerator::RandomGenerator(unsigned int seed) : m_engine(seed) {}

py::array_t<double> RandomGenerator::uniform(const unsigned int nrows, const unsigned int ncols) {
  std::uniform_real_distribution<double> dist{};
  arma::mat r(nrows, ncols, arma::fill::none);
  r.imbue([&]() { return dist(m_engine); });
  return carma::mat_to_arr(r, true);
}
