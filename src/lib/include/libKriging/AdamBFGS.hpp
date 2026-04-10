#ifndef LIBKRIGING_ADAM_BFGS_HPP
#define LIBKRIGING_ADAM_BFGS_HPP

#include "lbfgsb_cpp/lbfgsb.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <functional>
#include <vector>

/**
 * @brief Adam+BFGS bi-level optimizer.
 *
 * Splits a parameter vector x = [x_outer ; x_inner] into two groups:
 *   - x_outer (size n_outer): optimized with Adam (gradient ascent/descent)
 *   - x_inner (size n_inner): optimized with L-BFGS-B for each outer step
 *
 * The objective function f(x_outer, x_inner) is minimized (or maximized,
 * depending on the sign convention of the callback).
 *
 * This is the pattern used by WarpKriging where:
 *   - x_outer = warping parameters (optimized by Adam)
 *   - x_inner = log(theta) range parameters (optimized by L-BFGS)
 *
 * Usage:
 *   AdamBFGS opt(n_outer, n_inner);
 *   opt.max_iter_adam = 200;
 *   opt.adam_lr = 1e-3;
 *   opt.maximize = true;  // gradient ascent for log-likelihood
 *
 *   auto result = opt.optimize(
 *       x_outer0, x_inner0,
 *       inner_lower, inner_upper,
 *       // full objective: returns f and sets grad_outer, grad_inner
 *       [](const arma::vec& x_outer, const arma::vec& x_inner,
 *          arma::vec* grad_outer, arma::vec* grad_inner) -> double { ... }
 *   );
 */
class AdamBFGS {
 public:
  struct Result {
    arma::vec x_outer;
    arma::vec x_inner;
    double f_opt;
    arma::uword n_adam_iters;
    arma::uword n_bfgs_evals;  // total inner BFGS function evaluations
  };

  /// Callback signature:
  ///   f(x_outer, x_inner, grad_outer*, grad_inner*) -> objective value
  /// If grad_outer is non-null, fill ∂f/∂x_outer.
  /// If grad_inner is non-null, fill ∂f/∂x_inner.
  using ObjFn = std::function<
      double(const arma::vec& x_outer, const arma::vec& x_inner, arma::vec* grad_outer, arma::vec* grad_inner)>;

  arma::uword n_outer;
  arma::uword n_inner;

  // Adam parameters
  arma::uword max_iter_adam = 200;
  double adam_lr = 1e-3;
  double adam_beta1 = 0.9;
  double adam_beta2 = 0.999;
  double adam_eps = 1e-8;

  // BFGS parameters (for inner loop)
  arma::uword max_iter_bfgs = 100;
  double bfgs_pgtol = 1e-6;
  double bfgs_factr = 1e7;

  /// If true, maximize the objective (Adam does gradient ascent, BFGS minimizes -f).
  bool maximize = false;

  AdamBFGS(arma::uword n_outer_, arma::uword n_inner_) : n_outer(n_outer_), n_inner(n_inner_) {}

  /**
   * @brief Run the bi-level optimization.
   *
   * @param x_outer0     initial outer parameters
   * @param x_inner0     initial inner parameters
   * @param inner_lower  lower bounds for x_inner (can be empty for unbounded)
   * @param inner_upper  upper bounds for x_inner (can be empty for unbounded)
   * @param fn           objective function callback
   * @return Result with best parameters and objective value
   */
  Result optimize(arma::vec x_outer0,
                  arma::vec x_inner0,
                  const arma::vec& inner_lower,
                  const arma::vec& inner_upper,
                  ObjFn fn) const {
    arma::vec x_outer = std::move(x_outer0);
    arma::vec x_inner = std::move(x_inner0);

    // Adam moment accumulators
    arma::vec mm = arma::zeros(n_outer);
    arma::vec vm = arma::zeros(n_outer);

    double best_f = maximize ? -arma::datum::inf : arma::datum::inf;
    arma::vec best_outer = x_outer;
    arma::vec best_inner = x_inner;
    arma::uword total_bfgs_evals = 0;

    // Set up bounds for BFGS
    bool has_bounds = (inner_lower.n_elem == n_inner && inner_upper.n_elem == n_inner);
    arma::ivec bounds_type;
    if (has_bounds) {
      bounds_type.set_size(n_inner);
      bounds_type.fill(2);  // both lower and upper bounded
    }

    for (arma::uword t = 1; t <= max_iter_adam; ++t) {
      // ---- Inner loop: L-BFGS-B on x_inner for fixed x_outer ----
      if (n_inner > 0) {
        lbfgsb::Optimizer optimizer{static_cast<unsigned int>(n_inner)};
        optimizer.iprint = -1;
        optimizer.max_iter = max_iter_bfgs;
        optimizer.pgtol = bfgs_pgtol;
        optimizer.factr = bfgs_factr;

        // BFGS always minimizes, so negate if we're maximizing
        const double sign = maximize ? -1.0 : 1.0;

        auto bfgs_fn = [&](const arma::vec& xi, arma::vec& grad_i) -> double {
          double f = fn(x_outer, xi, nullptr, &grad_i);
          grad_i *= sign;
          return sign * f;
        };

        if (has_bounds) {
          auto res
              = optimizer.minimize(bfgs_fn, x_inner, inner_lower.memptr(), inner_upper.memptr(), bounds_type.memptr());
          total_bfgs_evals += res.num_fun_calls;
        } else {
          // Unbounded: use bound_type = 0 (no bounds)
          arma::ivec no_bounds(n_inner, arma::fill::zeros);
          arma::vec dummy_lb(n_inner, arma::fill::zeros);
          arma::vec dummy_ub(n_inner, arma::fill::zeros);
          auto res = optimizer.minimize(bfgs_fn, x_inner, dummy_lb.memptr(), dummy_ub.memptr(), no_bounds.memptr());
          total_bfgs_evals += res.num_fun_calls;
        }
      }

      // ---- Evaluate objective and get outer gradient ----
      arma::vec grad_outer(n_outer);
      double f = fn(x_outer, x_inner, &grad_outer, nullptr);

      // Track best
      bool improved = maximize ? (f > best_f) : (f < best_f);
      if (improved) {
        best_f = f;
        best_outer = x_outer;
        best_inner = x_inner;
      }

      // ---- Adam update on x_outer ----
      if (n_outer > 0) {
        // For maximization, Adam ascends (positive gradient direction)
        // For minimization, Adam descends (negative gradient direction)
        const arma::vec& g = maximize ? grad_outer : -grad_outer;

        mm = adam_beta1 * mm + (1.0 - adam_beta1) * g;
        vm = adam_beta2 * vm + (1.0 - adam_beta2) * (g % g);
        arma::vec mh = mm / (1.0 - std::pow(adam_beta1, static_cast<double>(t)));
        arma::vec vh = vm / (1.0 - std::pow(adam_beta2, static_cast<double>(t)));
        x_outer += adam_lr * mh / (arma::sqrt(vh) + adam_eps);
      }
    }

    return {best_outer, best_inner, best_f, max_iter_adam, total_bfgs_evals};
  }
};

#endif  // LIBKRIGING_ADAM_BFGS_HPP
