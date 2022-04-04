// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <random>

#include "libKriging/utils/lk_armadillo.hpp"


// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT void Random::set_seed(const int seed) {
    engine.seed(seed);
};

std::function<double()> Random::runif = []() {

};

std::function<arma::vec(const int)> Random::runif_vec = [](const int n) {
    std::uniform_real_distribution<double> dist{};
    arma::vec r(n, arma::fill::none);
    r.imbue([&]() { return dist(engine); });
    return r;
};

std::function<arma::mat(const int,const int)> Random::runif_mat = [](const int n, const int m) {
    std::uniform_real_distribution<double> dist{};
    arma::mat r(n, m, arma::fill::none);
    r.imbue([&]() { return dist(engine); });
    return r;
};

