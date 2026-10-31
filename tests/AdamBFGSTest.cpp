// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/AdamBFGS.hpp"
#include <cmath>
// clang-format on

/**
 * @file AdamBFGSTest.cpp
 * @brief The two safeguards of the bi-level Adam+BFGS optimizer used by
 *        WarpKriging (docs/dev/CholJitter.md, Adam phase):
 *  - restart of the first inner L-BFGS-B when it stops in <= 2 iterations
 *    (a corner of the bounds where the gradient points outward);
 *  - step decay at the plateaus of the Adam loop instead of stopping.
 */

TEST_CASE("AdamBFGS restarts the first inner BFGS stopped at a corner of the bounds", "[AdamBFGS]") {
  // Inner objective to maximize on [-3, 3]: increasing at the start x = 1 up
  // to the upper bound (a KKT point), with a higher interior maximum at -1.45
  auto fn = [](const arma::vec&, const arma::vec& xi, arma::vec*, arma::vec* grad_inner) -> double {
    const double x = xi(0);
    const double bump = 10.0 * std::exp(-(x + 1.5) * (x + 1.5));
    if (grad_inner)
      *grad_inner = arma::vec{1.0 - 2.0 * (x + 1.5) * bump};
    return x + bump;
  };
  const arma::vec lower{-3.0}, upper{3.0};

  AdamBFGS opt(0, 1);
  opt.maximize = true;
  opt.max_iter_adam = 3;

  auto plain = opt.optimize(arma::vec(), arma::vec{1.0}, lower, upper, fn);
  CHECK(plain.x_inner(0) == Approx(3.0));

  int restarts = 0;
  opt.inner_restart = [&restarts](const lbfgsb::OptimizeResult& res, const arma::vec&, const arma::vec&) {
    return (res.num_iters <= 2 && restarts++ < 2) ? arma::vec{-1.0} : arma::vec();
  };
  auto restarted = opt.optimize(arma::vec(), arma::vec{1.0}, lower, upper, fn);
  CHECK(restarted.x_inner(0) == Approx(-1.45).margin(0.01));
  CHECK(restarted.f_opt > plain.f_opt);
  CHECK(restarts == 1);
}

TEST_CASE("AdamBFGS step decay goes closer to the optimum than stopping at the plateau", "[AdamBFGS]") {
  // Outer objective to maximize: a fixed Adam step oscillates around 0.35
  auto fn = [](const arma::vec& xo, const arma::vec&, arma::vec* grad_outer, arma::vec*) -> double {
    const double u = xo(0) - 0.35;
    if (grad_outer)
      *grad_outer = arma::vec{-2.0 * u};
    return -u * u;
  };

  AdamBFGS opt(1, 0);
  opt.maximize = true;
  opt.adam_lr = 0.1;
  opt.max_iter_adam = 500;
  opt.patience = 5;
  opt.rel_tol = 1e-12;

  auto stopped = opt.optimize(arma::vec{0.0}, arma::vec(), arma::vec(), arma::vec(), fn);
  opt.lr_decays = 2;
  auto decayed = opt.optimize(arma::vec{0.0}, arma::vec(), arma::vec(), arma::vec(), fn);

  INFO("stopped: " << stopped.x_outer(0) << " after " << stopped.n_adam_iters << ", decayed: " << decayed.x_outer(0)
                   << " after " << decayed.n_adam_iters);
  CHECK(decayed.f_opt >= stopped.f_opt);
  CHECK(decayed.n_adam_iters > stopped.n_adam_iters);
  CHECK(std::abs(decayed.x_outer(0) - 0.35) < std::abs(stopped.x_outer(0) - 0.35));
}
