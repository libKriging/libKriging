// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <random>

#include "libKriging/Random.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

unsigned int Random::seed = 123;

std::mt19937 Random::engine(123);

LIBKRIGING_EXPORT void Random::reset_seed(unsigned int seed) {
  Random::seed = seed;
  Random::init();
};

void Random::init() {
  engine.seed(Random::seed);
};

LIBKRIGING_EXPORT std::function<float()> Random::randu = []() {
  std::uniform_real_distribution<float> dist{};
  return dist(engine);
};

LIBKRIGING_EXPORT std::function<arma::vec(const int)> Random::randu_vec = [](const int n) {
  std::uniform_real_distribution<float> dist{};
  arma::vec r(n, arma::fill::none);
  r.imbue([&]() { return dist(engine); });
  return r;
};

LIBKRIGING_EXPORT std::function<arma::mat(const int, const int)> Random::randu_mat = [](const int n, const int m) {
  std::uniform_real_distribution<float> dist{};
  arma::mat r(n, m, arma::fill::none);
  r.imbue([&]() { return dist(engine); });
  return r;
};

LIBKRIGING_EXPORT std::function<arma::mat(const int, const int)> Random::randn_mat = [](const int n, const int m) {
  std::normal_distribution<float> dist{};
  arma::mat r(n, m, arma::fill::none);
  r.imbue([&]() { return dist(engine); });
  return r;
};