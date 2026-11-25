// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Parallel.hpp"
#include <stdexcept>
// clang-format on

// The settings of docs/Settings.md (#468): each user setting has a setter and a
// getter; its LK_* environment variable only gives the default, read once when
// the library is loaded, and a setter call overrides it.

namespace {
struct Restore {
  bool reparam = Optim::is_reparametrized();
  double lower = Optim::get_theta_lower_factor();
  double upper = Optim::get_theta_upper_factor();
  bool variogram = Optim::variogram_bounds_heuristic_used();
  int log_level = Optim::get_log_level();
  int max_restart = Optim::get_max_restart();
  int max_iteration = Optim::get_max_iteration();
  double gtol = Optim::get_gradient_tolerance();
  double ftol = Optim::get_objective_rel_tolerance();
  double percent = Parallel::get_num_threads_percent();
  int pool = Parallel::get_thread_pool_size();
  int delay = Parallel::get_thread_start_delay_ms();
  int blas_min_n = Parallel::get_blas_threads_min_n();
  int omp_min_rows = Parallel::get_omp_min_rows();
  double jitter = LinearAlgebra::get_chol_jitter();
  ~Restore() {
    Optim::use_reparametrize(reparam);
    Optim::set_theta_lower_factor(lower);
    Optim::set_theta_upper_factor(upper);
    Optim::use_variogram_bounds_heuristic(variogram);
    Optim::set_log_level(log_level);
    Optim::set_max_restart(max_restart);
    Optim::set_max_iteration(max_iteration);
    Optim::set_gradient_tolerance(gtol);
    Optim::set_objective_rel_tolerance(ftol);
    Parallel::set_num_threads_percent(percent);
    Parallel::set_thread_pool_size(pool);
    Parallel::set_thread_start_delay_ms(delay);
    Parallel::set_blas_threads_min_n(blas_min_n);
    Parallel::set_omp_min_rows(omp_min_rows);
    LinearAlgebra::set_chol_jitter(jitter);
  }
};
}  // namespace

TEST_CASE("Settings: documented defaults without LK_* environment", "[settings]") {
  // only meaningful when none of the variables is set (ctest does not set them
  // for this case); a variable set by the caller would change the default
  if (std::getenv("LK_MAX_RESTART") == nullptr)
    CHECK(Optim::get_max_restart() == 10);
  if (std::getenv("LK_MAX_ITERATION") == nullptr)
    CHECK(Optim::get_max_iteration() == 20);
  if (std::getenv("LK_REPARAMETRIZE") == nullptr)
    CHECK(Optim::is_reparametrized());
  if (std::getenv("LK_LOG_LEVEL") == nullptr)
    CHECK(Optim::get_log_level() == 0);
  if (std::getenv("LK_CHOL_JITTER") == nullptr)
    CHECK(LinearAlgebra::get_chol_jitter() < 0.0);
}

TEST_CASE("Settings: set / get round trip of every user setting", "[settings]") {
  Restore restore;

  Optim::use_reparametrize(false);
  CHECK_FALSE(Optim::is_reparametrized());
  Optim::set_theta_lower_factor(0.25);
  CHECK(Optim::get_theta_lower_factor() == 0.25);
  Optim::set_theta_upper_factor(4.5);
  CHECK(Optim::get_theta_upper_factor() == 4.5);
  Optim::use_variogram_bounds_heuristic(false);
  CHECK_FALSE(Optim::variogram_bounds_heuristic_used());
  Optim::set_log_level(Optim::log_warning);
  CHECK(Optim::get_log_level() == Optim::log_warning);
  Optim::set_max_restart(3);
  CHECK(Optim::get_max_restart() == 3);
  Optim::set_max_restart(0);
  CHECK(Optim::get_max_restart() == 0);
  CHECK_THROWS_AS(Optim::set_max_restart(-1), std::invalid_argument);
  CHECK(Optim::get_max_restart() == 0);
  Optim::set_max_iteration(57);
  CHECK(Optim::get_max_iteration() == 57);
  Optim::set_gradient_tolerance(0.0125);
  CHECK(Optim::get_gradient_tolerance() == 0.0125);
  Optim::set_objective_rel_tolerance(0.0025);
  CHECK(Optim::get_objective_rel_tolerance() == 0.0025);
  Parallel::set_num_threads_percent(0.5);
  CHECK(Parallel::get_num_threads_percent() == 0.5);
  Parallel::set_thread_pool_size(2);
  CHECK(Parallel::get_thread_pool_size() == 2);
  Parallel::set_thread_start_delay_ms(7);
  CHECK(Parallel::get_thread_start_delay_ms() == 7);
  Parallel::set_blas_threads_min_n(250);
  CHECK(Parallel::get_blas_threads_min_n() == 250);
  Parallel::set_omp_min_rows(300);
  CHECK(Parallel::get_omp_min_rows() == 300);
  LinearAlgebra::set_chol_jitter(1e-7);
  CHECK(LinearAlgebra::get_chol_jitter() == 1e-7);
}

// Hidden case, run by ctest with every LK_* user variable set
// (tests/CMakeLists.txt, SettingsTest/LK_environment): the environment gives the
// value at load, a setter overrides it.
TEST_CASE("Settings: LK_* environment read at load, setter overrides it", "[.env]") {
  Restore restore;

  CHECK_FALSE(Optim::is_reparametrized());  // LK_REPARAMETRIZE=0
  CHECK(Optim::get_theta_lower_factor() == 0.05);
  CHECK(Optim::get_theta_upper_factor() == 20.0);
  CHECK_FALSE(Optim::variogram_bounds_heuristic_used());  // LK_VARIOGRAM_BOUNDS_HEURISTIC=false
  CHECK(Optim::get_log_level() == 1);
  CHECK(Optim::get_max_restart() == 4);
  CHECK(Optim::get_max_iteration() == 33);
  CHECK(Optim::get_gradient_tolerance() == 0.002);
  CHECK(Optim::get_objective_rel_tolerance() == 0.004);
  CHECK(Parallel::get_num_threads_percent() == 0.75);
  CHECK(Parallel::get_thread_pool_size() == 3);
  CHECK(Parallel::get_thread_start_delay_ms() == 5);
  CHECK(Parallel::get_blas_threads_min_n() == 123);
  CHECK(Parallel::get_omp_min_rows() == 456);
  CHECK(LinearAlgebra::get_chol_jitter() == 1e-8);

  // the setter wins over the environment
  Optim::set_max_restart(7);
  CHECK(Optim::get_max_restart() == 7);
  Optim::set_max_iteration(21);
  CHECK(Optim::get_max_iteration() == 21);
  Parallel::set_omp_min_rows(789);
  CHECK(Parallel::get_omp_min_rows() == 789);
  LinearAlgebra::set_chol_jitter(0.0);
  CHECK(LinearAlgebra::get_chol_jitter() == 0.0);
}
