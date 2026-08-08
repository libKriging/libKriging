// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

static double f2d(double x1, double x2) {
  return std::sin(3.0 * x1) + std::cos(5.0 * x2) + x1 * x2;
}

static void make_data(arma::uword n, arma::mat& X, arma::vec& y, unsigned seed = 123) {
  arma::arma_rng::set_seed(seed);
  X = arma::mat(n, 2, arma::fill::randu);
  y = arma::vec(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f2d(X(i, 0), X(i, 1));
}

static bool all_unique_and_in_range(const arma::uvec& idx, arma::uword n) {
  if (idx.n_elem != arma::uvec(arma::unique(idx)).n_elem)
    return false;
  return idx.is_empty() || idx.max() < n;
}

// -----------------------------------------------------------------------------

TEST_CASE("subsetOfData returns all indices when n_max >= n", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);

  arma::uvec idx = Kriging::subsetOfData(X, 30);
  CHECK(idx.n_elem == 30);
  CHECK(arma::all(idx == arma::regspace<arma::uvec>(0, 29)));

  arma::uvec idx2 = Kriging::subsetOfData(X, 100);
  CHECK(idx2.n_elem == 30);
}

TEST_CASE("subsetOfData returns empty for n_max = 0", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  CHECK(Kriging::subsetOfData(X, 0).n_elem == 0);
}

TEST_CASE("subsetOfData (kmeans) returns n_max unique valid indices", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  arma::uvec idx = Kriging::subsetOfData(X, 40, "kmeans");
  CHECK(idx.n_elem == 40);
  CHECK(all_unique_and_in_range(idx, 200));
  // sorted, as documented
  CHECK(arma::all(idx == arma::sort(idx)));
}

TEST_CASE("subsetOfData (random) returns n_max unique valid indices", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  arma::uvec idx = Kriging::subsetOfData(X, 40, "random");
  CHECK(idx.n_elem == 40);
  CHECK(all_unique_and_in_range(idx, 200));
}

TEST_CASE("subsetOfData is reproducible given the same seed", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  arma::uvec idx1 = Kriging::subsetOfData(X, 30, "kmeans", 42);
  arma::uvec idx2 = Kriging::subsetOfData(X, 30, "kmeans", 42);
  CHECK(arma::all(idx1 == idx2));

  arma::uvec idx3 = Kriging::subsetOfData(X, 30, "random", 42);
  arma::uvec idx4 = Kriging::subsetOfData(X, 30, "random", 42);
  CHECK(arma::all(idx3 == idx4));
}

TEST_CASE("subsetOfData rejects an unknown method", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  CHECK_THROWS_AS(Kriging::subsetOfData(X, 10, "bogus"), std::invalid_argument);
}

TEST_CASE("subsetOfData (kmeans) covers the domain better than a clustered baseline", "[subset][kriging]") {
  // Coverage = fill-distance: for every point of the FULL design, its
  // distance to the nearest SUBSET point; the worst (max) such distance
  // over the full design measures how well the subset covers the whole
  // domain. kmeans should leave much smaller gaps than an adversarial
  // "first n_max rows in a spatially-sorted order" baseline (clustered in
  // one coordinate, so it leaves the rest of the domain uncovered) -- NOT
  // the same thing as nearest-neighbor spacing within the subset itself
  // (a spread-out-over-the-whole-domain subset can have LARGER intra-subset
  // gaps than a locally-clustered one while still covering the domain far
  // better, which is what actually matters for a representative subset).
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);
  // sort rows by first coordinate so "first n_max rows" is a clustered,
  // non-representative baseline
  arma::uvec order = arma::sort_index(X.col(0));
  arma::mat X_sorted = X.rows(order);

  arma::uvec idx = Kriging::subsetOfData(X_sorted, 30, "kmeans");
  arma::mat X_sub = X_sorted.rows(idx);
  arma::mat X_first30 = X_sorted.rows(arma::regspace<arma::uvec>(0, 29));

  auto fill_distance = [&X_sorted](const arma::mat& subset) {
    double worst = 0.0;
    for (arma::uword i = 0; i < X_sorted.n_rows; ++i) {
      double best = arma::datum::inf;
      for (arma::uword j = 0; j < subset.n_rows; ++j)
        best = std::min(best, arma::norm(X_sorted.row(i) - subset.row(j)));
      worst = std::max(worst, best);
    }
    return worst;
  };

  const double fill_kmeans = fill_distance(X_sub);
  const double fill_first30 = fill_distance(X_first30);
  INFO("fill-distance (worst-covered point): kmeans subset = " << fill_kmeans
                                                                << ", first-30 baseline = " << fill_first30);
  CHECK(fill_kmeans < fill_first30);
}

TEST_CASE("fitting on a subset gives sane (not wildly degraded) predictions", "[subset][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(30, Xt, yt, 456);

  Kriging k_full(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  auto [m_full, s_full, c1, d1, d2] = k_full.predict(Xt, true, false, false);
  const double rmse_full = std::sqrt(arma::mean(arma::square(m_full - yt)));

  arma::uvec idx = Kriging::subsetOfData(X, 60, "kmeans");
  Kriging k_sub(y(idx), X.rows(idx), "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  auto [m_sub, s_sub, c3, d3, d4] = k_sub.predict(Xt, true, false, false);
  const double rmse_sub = std::sqrt(arma::mean(arma::square(m_sub - yt)));

  const double sdy = arma::stddev(y);
  INFO("rmse_full=" << rmse_full << " rmse_sub(n=60/200)=" << rmse_sub << " sd(y)=" << sdy);
  // A 60/200 subset should still predict a smooth-ish function reasonably
  // (well within one output standard deviation), even if visibly worse than
  // using the full data.
  CHECK(rmse_sub < sdy);
}
