#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Optim {
 public:
  static bool reparametrize;
  LIBKRIGING_EXPORT static void use_reparametrize(bool do_reparametrize);
  LIBKRIGING_EXPORT static bool is_reparametrized();
  static std::function<double(const double&)> reparam_to_;
  static std::function<arma::vec(const arma::vec&)> reparam_to;
  static std::function<double(const double&)> reparam_from_;
  static std::function<arma::vec(const arma::vec&)> reparam_from;
  static std::function<double(const double&, const double&)> reparam_from_deriv_;
  static std::function<arma::vec(const arma::vec&, const arma::vec&)> reparam_from_deriv;
  static std::function<arma::mat(const arma::vec&, const arma::vec&, const arma::mat&)> reparam_from_deriv2;

  static double theta_lower_factor;
  LIBKRIGING_EXPORT static void set_theta_lower_factor(double _theta_lower_factor);
  LIBKRIGING_EXPORT static double get_theta_lower_factor();

  static double theta_upper_factor;
  LIBKRIGING_EXPORT static void set_theta_upper_factor(double _theta_upper_factor);
  LIBKRIGING_EXPORT static double get_theta_upper_factor();

  static bool variogram_bounds_heuristic;
  LIBKRIGING_EXPORT static void use_variogram_bounds_heuristic(bool _variogram_bounds_heuristic);
  LIBKRIGING_EXPORT static bool variogram_bounds_heuristic_used();

  static int log_level;
  LIBKRIGING_EXPORT static void log(int t);

  static int max_restart;  // eg. for wrong convergence to bounds

  static int max_iteration;
  LIBKRIGING_EXPORT static void set_max_iteration(int max_iteration_val);
  LIBKRIGING_EXPORT static int get_max_iteration();

  static double gradient_tolerance;
  LIBKRIGING_EXPORT static void set_gradient_tolerance(double gradient_tolerance_val);
  LIBKRIGING_EXPORT static double get_gradient_tolerance();

  static double objective_rel_tolerance;
  LIBKRIGING_EXPORT static void set_objective_rel_tolerance(double objective_rel_tolerance_val);
  LIBKRIGING_EXPORT static double get_objective_rel_tolerance();

  static bool quadfailover;
  LIBKRIGING_EXPORT static void use_quadfailover(bool do_quadfailover);
  LIBKRIGING_EXPORT static bool is_quadfailover();
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINLIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPPEARALGEBRA_HPP
