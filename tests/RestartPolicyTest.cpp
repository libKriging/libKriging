// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "RestartPolicy.hpp"
// clang-format on

/**
 * @file RestartPolicyTest.cpp
 * @brief Restart policy of the L-BFGS-B runs shared by Kriging, WarpKriging
 *        (joint BFGS) and the inner BFGS of AdamBFGS
 *        (docs/dev/rationalization/T2-restart-policy.md).
 */

using libKriging::detail::RestartPolicy;

namespace {
const std::string converged = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH";
const std::string abnormal = "ABNORMAL_TERMINATION_IN_LNSRCH";
}  // namespace

TEST_CASE("RestartPolicy criteria", "[RestartPolicy]") {
  RestartPolicy::Config config;
  config.max_restart = 10;
  config.factr = 1e7;
  config.worse_criterion = true;

  SECTION("a converged run is not restarted") {
    RestartPolicy p(config);
    p.record(1.0);
    CHECK_FALSE(p.restart(converged, 10, false));
    CHECK(p.retry() == 0);
  }
  SECTION("abnormal stop, <= 2 iterations, lower bound") {
    RestartPolicy p(config);
    p.record(1.0);
    CHECK(p.restart(abnormal, 10, false));
    CHECK(p.last_reasons() == RestartPolicy::Abnormal);
    p.record(1.0);
    CHECK(p.restart(converged, 2, false));
    CHECK(p.last_reasons() == RestartPolicy::FewIterations);
    p.record(1.0);
    CHECK(p.restart(converged, 10, true));
    CHECK(p.last_reasons() == RestartPolicy::AtLowerBound);
    CHECK(p.retry() == 3);
  }
  SECTION("at most max_restart restarts") {
    RestartPolicy p(config);
    int n = 0;
    while (p.record(1.0), p.restart(abnormal, 10, false))
      n++;
    CHECK(n == 10);
  }
  SECTION("long range: at most max_long_range_restart") {
    config.max_long_range_restart = 2;
    RestartPolicy p(config);
    int n = 0;
    while (p.record(1.0), p.restart(converged, 10, false, /*long_range=*/true))
      n++;
    CHECK(n == 2);
    CHECK_FALSE(p.long_range_allowed());
  }
  SECTION("worse within factr * epsmch is no change") {
    RestartPolicy p(config);
    p.record(100.0);
    REQUIRE(p.restart(abnormal, 10, false));
    p.record(100.0 * (1 + 0.5 * config.factr * arma::datum::eps));
    CHECK_FALSE(p.restart(converged, 10, false));
  }
  SECTION("a worse restart does not trigger another one by itself") {
    RestartPolicy p(config);
    CHECK(p.record(1.0));
    REQUIRE(p.restart(abnormal, 10, false));
    CHECK_FALSE(p.record(2.0));
    REQUIRE(p.restart(converged, 10, false));  // worse than the first run
    CHECK(p.last_reasons() == RestartPolicy::Worse);
    p.record(3.0);
    CHECK_FALSE(p.restart(converged, 10, false));  // worse again: no cascade
    CHECK(p.retry() == 2);
    CHECK(p.best() == 1.0);
  }
  SECTION("a worse run restarted by another criterion still counts") {
    RestartPolicy p(config);
    p.record(1.0);
    REQUIRE(p.restart(abnormal, 10, false));
    p.record(2.0);
    REQUIRE(p.restart(converged, 10, false));  // worse
    p.record(3.0);
    REQUIRE(p.restart(abnormal, 10, false));  // abnormal; worse ignored (worse-only restart)
    CHECK(p.last_reasons() == RestartPolicy::Abnormal);
    p.record(4.0);
    CHECK(p.restart(converged, 10, false));  // worse, after a non-worse-only restart
  }
  SECTION("worse criterion off") {
    config.worse_criterion = false;
    RestartPolicy p(config);
    p.record(1.0);
    REQUIRE(p.restart(abnormal, 10, false));
    p.record(2.0);
    CHECK_FALSE(p.restart(converged, 10, false));
  }
}

TEST_CASE("RestartPolicy restart point and lower bound distance", "[RestartPolicy]") {
  RestartPolicy::Config config;
  config.max_restart = 10;
  RestartPolicy p(config);
  const arma::vec lower{0.1, 0.1}, upper{10.0, 10.0}, start{1.0, 0.2};
  p.record(1.0);
  REQUIRE(p.restart(abnormal, 10, false));
  arma::vec t1 = p.restart_theta(start, lower, upper);
  CHECK(t1(0) == Approx(0.55));
  CHECK(t1(1) == Approx(0.15));
  p.record(1.0);
  REQUIRE(p.restart(abnormal, 10, false));
  arma::vec t2 = p.restart_theta(start, lower, upper);
  CHECK(t2(0) == Approx(0.275));
  CHECK(t2(1) == lower(1));  // (0.2 + 0.1) / 4 < 0.1: kept inside the bounds

  CHECK(RestartPolicy::distance_to_lower(arma::vec{0.1, 5.0}, lower, upper) == 0.0);
  // pinned dimension (lower == upper) ignored
  CHECK(RestartPolicy::distance_to_lower(arma::vec{0.1, 5.0}, lower, arma::vec{0.1, 10.0}) == Approx(4.9));
  CHECK(RestartPolicy::distance_to_lower(arma::vec{0.1}, arma::vec{0.1}, arma::vec{0.1}) == arma::datum::inf);
}
