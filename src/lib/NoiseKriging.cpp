// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NoiseKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
#include <tuple>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

// Weak symbol for OpenBLAS thread control (if available)
extern "C" {
  void openblas_set_num_threads(int num_threads) __attribute__((weak));
}

/************************************************/
/**      NoiseKriging implementation        **/
/************************************************/

// This will create the dist(xi,xj) function above. Need to parse "covType".
void NoiseKriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    _Cov = Covariance::Cov_gauss;
    _DlnCovDtheta = Covariance::DlnCovDtheta_gauss;
    _DlnCovDx = Covariance::DlnCovDx_gauss;
    _Cov_pow = 2;
  } else if (covType.compare("exp") == 0) {
    _Cov = Covariance::Cov_exp;
    _DlnCovDtheta = Covariance::DlnCovDtheta_exp;
    _DlnCovDx = Covariance::DlnCovDx_exp;
    _Cov_pow = 1;
  } else if (covType.compare("matern3_2") == 0) {
    _Cov = Covariance::Cov_matern32;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern32;
    _DlnCovDx = Covariance::DlnCovDx_matern32;
    _Cov_pow = 1.5;
  } else if (covType.compare("matern5_2") == 0) {
    _Cov = Covariance::Cov_matern52;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern52;
    _DlnCovDx = Covariance::DlnCovDx_matern52;
    _Cov_pow = 2.5;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

LIBKRIGING_EXPORT arma::mat NoiseKriging::covMat(const arma::mat& X1, const arma::mat& X2) {
  arma::mat Xn1 = X1;
  arma::mat Xn2 = X2;
  Xn1.each_row() -= m_centerX;
  Xn1.each_row() /= m_scaleX;
  Xn2.each_row() -= m_centerX;
  Xn2.each_row() /= m_scaleX;

  arma::mat R = arma::mat(X1.n_rows, X2.n_rows, arma::fill::none);
  LinearAlgebra::covMat_rect(&R, Xn1.t(), Xn2.t(), m_theta, _Cov, m_sigma2);
  return R;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const arma::vec& y,
                                             const arma::vec& noise,
                                             const arma::mat& X,
                                             const std::string& covType,
                                             const Trend::RegressionModel& regmodel,
                                             bool normalize,
                                             const std::string& optim,
                                             const std::string& objective,
                                             const Parameters& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X.n_rows) + "x"
                             + std::to_string(X.n_cols) + "), y: (" + std::to_string(y.n_elem) + ")");

  make_Cov(covType);
  fit(y, noise, X, regmodel, normalize, optim, objective, parameters);
}

LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const NoiseKriging& other, ExplicitCopySpecifier) : NoiseKriging{other} {}

void NoiseKriging::populate_Model(KModel& m,
                                  const arma::vec& theta,
                                  const double sigma2,
                                  std::map<std::string, double>* bench) const {
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::uword p = m_F.n_cols;

  auto t0 = Bench::tic();
  // Reuse existing m.R allocation
  // check if we want to recompute model for same theta, for augmented Xy (using cholesky fast update).
  bool update = false;
  if (!m_is_empty)
    update = (m_sigma2 == sigma2) && (m_theta.size() == theta.size()) && (theta - m_theta).is_zero()
             && (this->m_T.memptr() != nullptr) && (n > this->m_T.n_rows);
  if (update) {
    m.L = LinearAlgebra::update_cholCov(&(m.R), m_dX, theta, _Cov, sigma2, sigma2 + m_noise, m_T, m_R);
  } else
    m.L = LinearAlgebra::cholCov(&(m.R), m_dX, theta, _Cov, sigma2, sigma2 + m_noise);
  t0 = Bench::toc(bench, "R = _Cov(dX) & L = Chol(R)", t0);

  // Compute intermediate useful matrices
  arma::mat Fystar = LinearAlgebra::solve(m.L, arma::join_rows(m_F, m_y));
  t0 = Bench::toc(bench, "Fy* = L \\ [F,y]", t0);
  m.Fstar = Fystar.head_cols(p);
  m.ystar = Fystar.tail_cols(1);

  arma::mat Q_qr;
  arma::mat R_qr;
  arma::qr_econ(Q_qr, R_qr, Fystar);
  t0 = Bench::toc(bench, "Q_qr,R_qr = QR(Fy*)", t0);

  m.Rstar = R_qr.head_cols(p);
  m.Qstar = Q_qr.head_cols(p);
  m.Estar = Q_qr.tail_cols(1) * R_qr.at(p, p);
  m.SSEstar = R_qr.at(p, p) * R_qr.at(p, p);

  if (m_est_beta) {
    m.betahat = LinearAlgebra::solve(m.Rstar, R_qr.tail_cols(1));
    t0 = Bench::toc(bench, "^b = R* \\ R_qr[1:p, p+1]", t0);
  } else {
    m.betahat = arma::vec(p, arma::fill::zeros);  // whatever: not used
  }
}

NoiseKriging::KModel NoiseKriging::make_Model(const arma::vec& theta,
                                              const double sigma2,
                                              std::map<std::string, double>* bench) const {
  arma::mat R;
  arma::mat L;
  arma::mat Linv;
  arma::mat Fstar;
  arma::vec ystar;
  arma::mat Rstar;
  arma::mat Qstar;
  arma::vec Estar;
  double SSEstar{};
  arma::vec betahat;
  NoiseKriging::KModel m{R, L, Linv, Fstar, ystar, Rstar, Qstar, Estar, SSEstar, betahat};

  arma::uword n = m_X.n_rows;
  arma::uword p = m_F.n_cols;

  // Allocate matrices
  m.R = arma::mat(n, n, arma::fill::none);
  m.L = arma::mat(n, n, arma::fill::none);
  m.Linv = arma::mat();  // Empty matrix, will be filled on demand in gradient computation
  m.Fstar = arma::mat(n, p, arma::fill::none);
  m.ystar = arma::vec(n, arma::fill::none);
  m.Rstar = arma::mat(p, p, arma::fill::none);
  m.Qstar = arma::mat(n, p, arma::fill::none);
  m.Estar = arma::vec(n, arma::fill::none);
  m.betahat = arma::vec(p, arma::fill::none);

  // Populate the model
  populate_Model(m, theta, sigma2, bench);

  return m;
}

// Objective function for fit : -logLikelihood

double NoiseKriging::_logLikelihood(const arma::vec& _theta_sigma2,
                                    arma::vec* grad_out,
                                    NoiseKriging::KModel* model,
                                    std::map<std::string, double>* bench) const {
  // arma::cout << " theta, sigma2: " << _theta_sigma2.t() << arma::endl;

  arma::uword d = m_X.n_cols;
  double _sigma2 = _theta_sigma2.at(d);
  if (!m_est_sigma2) {  // Force sigma2 to fixed value, if defined
    _sigma2 = m_sigma2;
  }
  arma::vec _theta = _theta_sigma2.head(d);

  NoiseKriging::KModel m = make_Model(_theta, _sigma2, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;

  // this L matrix is already multplied by sigma2
  double ll = -0.5 * (n * log(2 * M_PI) + 2 * sum(log(m.L.diag())) + as_scalar(LinearAlgebra::crossprod(m.Estar)));

  if (grad_out != nullptr) {
    auto t0 = Bench::tic();
    arma::vec terme1 = arma::vec(d);

    if ((m.Linv.memptr() == nullptr) || (arma::size(m.Linv) != arma::size(m.L))) {
      m.Linv = LinearAlgebra::solve(m.L, arma::mat(n, n, arma::fill::eye));
      t0 = Bench::toc(bench, "L ^-1", t0);
    }

    arma::mat Cinv = LinearAlgebra::crossprod(m.Linv);
    t0 = Bench::toc(bench, "R^-1 = t(L^-1) * L^-1", t0);

    arma::mat x = LinearAlgebra::solve(m.L.t(), m.Estar);
    t0 = Bench::toc(bench, "x = tL \\ z", t0);

    arma::cube gradC = arma::cube(n, n, d, arma::fill::none);
    const arma::vec zeros = arma::vec(d, arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      gradC.tube(i, i) = zeros;
      for (arma::uword j = 0; j < i; j++) {
        gradC.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
        gradC.tube(j, i) = gradC.tube(i, j);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::mat gradC_k = gradC.slice(k);
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      terme1.at(k) = as_scalar(x.t() * gradC_k * x);
      double terme2 = -arma::trace(Cinv * gradC_k);
      (*grad_out).at(k) = (terme1.at(k) + terme2) / 2;
      t0 = Bench::toc(bench, "grad_ll[k] = xt * gradR_k / S2 + tr(Ri * gradR_k)", t0);
    }

    if (m_est_sigma2) {
      arma::mat dCdv = (m.R - arma::diagmat(m_noise)) / _sigma2;
      double _terme1 = -as_scalar((trans(x) * dCdv) * x);
      double _terme2 = arma::accu(arma::dot(Cinv, dCdv));
      (*grad_out).at(d) = -0.5 * (_terme1 + _terme2);
    } else
      (*grad_out).at(d) = 0;  // if sigma2 is defined & fixed by user

    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NoiseKriging::logLikelihoodFun(const arma::vec& _theta_sigma2,
                                                                               const bool _grad,
                                                                               const bool _bench) {
  double ll = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta_sigma2.n_elem);
      ll = _logLikelihood(_theta_sigma2, &grad, nullptr, &bench);
    } else
      ll = _logLikelihood(_theta_sigma2, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench) {
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;
    }

  } else {
    if (_grad) {
      grad = arma::vec(_theta_sigma2.n_elem);
      ll = _logLikelihood(_theta_sigma2, &grad, nullptr, nullptr);
    } else
      ll = _logLikelihood(_theta_sigma2, nullptr, nullptr, nullptr);
  }

  return std::make_tuple(ll, std::move(grad));
}

LIBKRIGING_EXPORT double NoiseKriging::logLikelihood() {
  int d = m_theta.n_elem;
  arma::vec _theta_sigma2 = arma::vec(d + 1);
  _theta_sigma2.head(d) = m_theta;
  _theta_sigma2.at(d) = m_sigma2;
  return std::get<0>(NoiseKriging::logLikelihoodFun(_theta_sigma2, false, false));
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void NoiseKriging::fit(const arma::vec& y,
                                         const arma::vec& noise,
                                         const arma::mat& X,
                                         const Trend::RegressionModel& regmodel,
                                         bool normalize,
                                         const std::string& optim,
                                         const std::string& objective,
                                         const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data) {
        // Change variable for opt: . -> 1/exp(.)
        const arma::vec _theta_sigma2 = Optim::reparam_from(_gamma);
        double ll = this->_logLikelihood(_theta_sigma2, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -Optim::reparam_from_deriv(_theta_sigma2, *grad_out);
        }
        return -ll;
      };
    } else {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data) {
        const arma::vec _theta_sigma2 = _gamma;
        double ll = this->_logLikelihood(_theta_sigma2, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -*grad_out;
        }
        return -ll;
      };
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported is: LL)");

  arma::rowvec centerX;
  arma::rowvec scaleX;
  double centerY;
  double scaleY;
  // Normalization of inputs and output
  m_normalize = normalize;
  if (m_normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
  } else {
    centerX = arma::rowvec(d, arma::fill::zeros);
    scaleX = arma::rowvec(d, arma::fill::ones);
    centerY = 0;
    scaleY = 1;
  }
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  {  // FIXME why copies of newX and newy
    arma::mat newX = X;
    newX.each_row() -= centerX;
    newX.each_row() /= scaleX;
    arma::vec newy = (y - centerY) / scaleY;
    arma::vec newnoise = noise / (scaleY * scaleY);
    this->m_X = newX;
    this->m_y = newy;
    this->m_noise = newnoise;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  // Note: m_dX is transposed compared to m_X
  m_dX = LinearAlgebra::compute_dX(m_X);
  m_maxdX = arma::max(arma::abs(m_dX), 1);

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);
  m_est_beta = parameters.is_beta_estim && (m_regmodel != Trend::RegressionModel::None);
  if (!m_est_beta && parameters.beta.has_value()
      && parameters.beta.value().n_elem > 0) {  // Then force beta to be fixed (not estimated, no variance)
    m_est_beta = false;
    m_beta = parameters.beta.value();
    if (m_normalize)
      m_beta /= scaleY;
  } else
    m_est_beta = true;

  arma::mat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if ((theta0.n_cols != d) && (theta0.n_rows == d))
      theta0 = theta0.t();
    if (m_normalize)
      theta0.each_row() /= scaleX;
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(d) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
  }

  if (optim == "none") {  // just keep given theta, no optimisation of ll (but estim beta still possible)
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");
    if (!parameters.sigma2.has_value())
      throw std::runtime_error("Sigma2 should be given, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;

    m_sigma2 = parameters.sigma2.value()[0];
    if (m_normalize)
      m_sigma2 /= (scaleY * scaleY);
    m_est_sigma2 = false;

    NoiseKriging::KModel m = make_Model(m_theta, m_sigma2, nullptr);
    m_is_empty = false;
    m_T = std::move(m.L);
    m_R = std::move(m.R);
    m_M = std::move(m.Fstar);
    m_circ = std::move(m.Rstar);
    m_star = std::move(m.Qstar);
    if (m_est_beta) {
      m_beta = std::move(m.betahat);
      m_z = std::move(m.Estar);
    } else {
      // m_beta = parameters.beta.value(); already done above
      m_z = std::move(m.ystar) - m_M * m_beta;
    }

  } else if (optim.rfind("BFGS", 0) == 0) {
    Random::init();

    arma::vec theta_lower = Optim::theta_lower_factor * m_maxdX;
    arma::vec theta_upper = Optim::theta_upper_factor * m_maxdX;

    arma::vec dy2 = arma::vec(n * n, arma::fill::zeros);
    for (arma::uword ij = 0; ij < dy2.n_elem; ij++) {
      int i = (int)ij / n;
      int j = ij % n;  // i,j <-> i*n+j
      if (i < j) {
        dy2[ij] = m_y.at(i) - m_y.at(j);
        dy2[ij] *= dy2[ij];
        dy2[j * n + i] = dy2[ij];
      }
    }

    if (Optim::variogram_bounds_heuristic) {
      // dy2 /= arma::var(m_y);
      arma::vec dy2dX2_slope = dy2 / arma::sum(m_dX % m_dX, 0).t();
      // arma::cout << "dy2dX_slope:" << dy2dX_slope << arma::endl;
      dy2dX2_slope.replace(arma::datum::nan, 0.0);  // we are not interested in same points where dX=0, and dy=0
      arma::vec w = dy2dX2_slope / sum(dy2dX2_slope);
      arma::mat steepest_dX_mean = arma::abs(m_dX) * w;
      // arma::cout << "steepest_dX_mean:" << steepest_dX_mean << arma::endl;

      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor * steepest_dX_mean);
      // no, only relevant for inf bound: theta_upper = arma::min(theta_upper, Optim::theta_upper_factor *
      // steepest_dX_mean);
      theta_lower = arma::min(theta_lower, theta_upper);
      theta_upper = arma::max(theta_lower, theta_upper);
    }
    // arma::cout << "theta_lower:" << theta_lower << arma::endl;
    // arma::cout << "theta_upper:" << theta_upper << arma::endl;

    int multistart = 1;
    try {
      multistart = std::stoi(optim.substr(4));
    } catch (std::invalid_argument&) {
      // let multistart = 1
    }

    // Configure threads for Armadillo/BLAS to balance nested parallelism
    // Each of the 'multistart' threads will use internal parallelism
    unsigned int n_cpu = std::thread::hardware_concurrency();
    if (n_cpu > 0 && multistart > 1) {
      unsigned int threads_per_worker = std::max(1u, n_cpu / multistart);
      
      // Set OpenBLAS threads (if available)
      if (openblas_set_num_threads != nullptr) {
        openblas_set_num_threads(threads_per_worker);
      }
      
      // Set OpenMP threads (for Armadillo operations that use OpenMP)
      #ifdef _OPENMP
      omp_set_num_threads(threads_per_worker);
      #endif
      
      if (Optim::log_level > 0) {
        arma::cout << "Threads per worker: " << threads_per_worker 
                   << " (total CPUs: " << n_cpu << ", multistart: " << multistart << ")" << arma::endl;
      }
    }

    arma::mat theta0_rand
        = arma::repmat(trans(theta_lower), multistart, 1)
          + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
    // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
    //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);

    if (parameters.theta.has_value()) {  // just use given theta(s) as starting values for multi-bfgs
      multistart = std::max(multistart, (int)theta0.n_rows);
      theta0 = arma::join_cols(theta0, theta0_rand);  // append random starting points to given ones
      theta0.resize(multistart, theta0.n_cols);       // keep only multistart first rows
    } else {
      theta0 = theta0_rand;
    }
    // arma::cout << "theta0:" << theta0 << arma::endl;

    arma::vec dX2 = arma::sum(m_dX % m_dX, 0).t();

    // see https://github.com/cran/DiceKriging/blob/547135515e32fa0a37260b9cd01631c1b7a69a5b/R/kmNuggets.init.R#L30
    double sigma2_variogram = 0.5 * arma::mean(dy2.elem(arma::find(dX2 >= arma::median(dX2))));
    double sigma2_lower = 0.1 * (sigma2_variogram - arma::max(m_noise));
    double sigma2_upper = 10 * (sigma2_variogram - arma::min(m_noise));
    arma::vec sigma20;
    if (parameters.sigma2.has_value()) {
      sigma20 = arma::vec(parameters.sigma2.value());
      if (m_normalize)
        sigma20 /= scaleY;
    } else {
      sigma20 = sigma2_lower + (sigma2_upper - sigma2_lower) * Random::randu_vec(theta0.n_rows);
    }
    // arma::cout << "sigma20:" << sigma20 << arma::endl;

    arma::vec gamma_lower = arma::vec(d + 1);
    gamma_lower.head(d) = theta_lower;
    gamma_lower.at(d) = sigma2_lower;
    arma::vec gamma_upper = arma::vec(d + 1);
    gamma_upper.head(d) = theta_upper;
    gamma_upper.at(d) = sigma2_upper;
    if (Optim::reparametrize) {
      arma::vec gamma_lower_tmp = gamma_lower;
      gamma_lower = Optim::reparam_to(gamma_upper);
      gamma_upper = Optim::reparam_to(gamma_lower_tmp);
    }

    // Set estimation flags before parallel execution
    m_est_sigma2 = parameters.is_sigma2_estim;
    if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
      m_sigma2 = parameters.sigma2.value()[0];
      if (m_normalize)
        m_sigma2 /= (scaleY * scaleY);
    } else {
      m_est_sigma2 = true;  // force estim if no value given
    }

    double min_ofn = std::numeric_limits<double>::infinity();

    // Pre-allocate KModel structures (one per multistart)
    arma::uword n_data = n;
    arma::uword p_data = m_F.n_cols;
    std::vector<NoiseKriging::KModel> preallocated_models(multistart);
    
    if (Optim::log_level > 0) {
      arma::cout << "Preallocating " << multistart << " KModel structures (n=" 
                 << n_data << ", p=" << p_data << ")..." << arma::endl;
    }
    
    for (int i = 0; i < multistart; i++) {
      auto& m = preallocated_models[i];
      m.R = arma::mat(n_data, n_data, arma::fill::none);
      m.L = arma::mat(n_data, n_data, arma::fill::none);
      m.Linv = arma::mat();  // Empty matrix
      m.Fstar = arma::mat(n_data, p_data, arma::fill::none);
      m.ystar = arma::vec(n_data, arma::fill::none);
      m.Rstar = arma::mat(p_data, p_data, arma::fill::none);
      m.Qstar = arma::mat(n_data, p_data, arma::fill::none);
      m.Estar = arma::vec(n_data, arma::fill::none);
      m.betahat = arma::vec(p_data, arma::fill::none);
      m.SSEstar = 0.0;
    }

    // Prepare gamma bounds for all starts
    std::vector<arma::vec> all_gamma_lower(multistart);
    std::vector<arma::vec> all_gamma_upper(multistart);
    
    for (arma::uword i = 0; i < multistart; i++) {
      arma::vec gamma_tmp = arma::vec(d + 1);
      gamma_tmp.head(d) = theta0.row(i % multistart).t();
      gamma_tmp.at(d) = sigma20[i % sigma20.n_elem];
      if (Optim::reparametrize) {
        gamma_tmp = Optim::reparam_to(gamma_tmp);
      }

      all_gamma_lower[i] = arma::min(gamma_tmp, gamma_lower);
      all_gamma_upper[i] = arma::max(gamma_tmp, gamma_upper);
    }

    // Thread pool configuration will be handled in the parallel execution section below

    // Multi-threading implementation for BFGS multistart
    // Each thread uses its own preallocated KModel, so no mutex needed
    
    // Structure to hold optimization results from each thread
    struct OptimizationResult {
      arma::uword start_index;
      double objective_value;
      arma::vec gamma;
      arma::vec theta_sigma2;
      arma::mat L;
      arma::mat R;
      arma::mat Fstar;
      arma::mat Rstar;
      arma::mat Qstar;
      arma::vec Estar;
      arma::vec ystar;
      double SSEstar;
      arma::vec betahat;
      bool success;
      std::string error_message;
      
      OptimizationResult() : start_index(0), objective_value(std::numeric_limits<double>::infinity()), success(false) {}
    };
    
    // Worker function returns OptimizationResult
    auto optimize_worker = [&](arma::uword start_idx) -> OptimizationResult {
      OptimizationResult result;
      result.start_index = start_idx;
      
      try {
        arma::vec gamma_tmp = arma::vec(d + 1);
        gamma_tmp.head(d) = theta0.row(start_idx % multistart).t();
        gamma_tmp.at(d) = sigma20[start_idx % sigma20.n_elem];
        if (Optim::reparametrize) {
          gamma_tmp = Optim::reparam_to(gamma_tmp);
        }

        arma::vec gamma_lower_local = all_gamma_lower[start_idx];
        arma::vec gamma_upper_local = all_gamma_upper[start_idx];

        if (Optim::log_level > 0) {
          arma::cout << "BFGS (start " << (start_idx+1) << "/" << multistart << "):" << arma::endl;
          arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
          arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
          arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower.t() << "";
          arma::cout << "                " << sigma2_lower << arma::endl;
          arma::cout << "  upper_bounds: " << theta_upper.t() << "";
          arma::cout << "                " << sigma2_upper << arma::endl;
          arma::cout << "  start_point: " << theta0.row(start_idx % multistart) << "";
          arma::cout << "               " << sigma20[start_idx % sigma20.n_elem] << arma::endl;
        }

        // Use pre-allocated KModel for this thread (thread-safe)
        if (start_idx >= preallocated_models.size()) {
          throw std::runtime_error("Preallocated model index out of bounds");
        }
        
        NoiseKriging::KModel& m = preallocated_models[start_idx];
        populate_Model(m, theta0.row(start_idx % multistart).t(), sigma20[start_idx % sigma20.n_elem], nullptr);

        lbfgsb::Optimizer optimizer{d + 1};
        optimizer.iprint = -1;  // Suppress LBFGSB output (we handle logging)
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d + 1, arma::fill::value(2)};

        int retry = 0;
        double best_f_opt = std::numeric_limits<double>::infinity();
        arma::vec best_gamma = gamma_tmp;

        while (retry <= Optim::max_restart) {
          arma::vec gamma_0 = gamma_tmp;
          auto opt_result = optimizer.minimize(
              [&m, &fit_ofn](const arma::vec& vals_inp, arma::vec& grad_out) -> double {
                return fit_ofn(vals_inp, &grad_out, &m);
              },
              gamma_tmp,
              gamma_lower_local.memptr(),
              gamma_upper_local.memptr(),
              bounds_type.memptr());

          if (opt_result.f_opt < best_f_opt) {
            best_f_opt = opt_result.f_opt;
            best_gamma = gamma_tmp;
          }

          double sol_to_lb_theta = arma::min(arma::abs(gamma_tmp.head(d) - gamma_lower_local.head(d)));
          double sol_to_ub_theta = arma::min(arma::abs(gamma_tmp.head(d) - gamma_upper_local.head(d)));
          double sol_to_b_theta = std::min(sol_to_ub_theta, sol_to_lb_theta);
          double sol_to_lb_sigma2 = std::abs(gamma_tmp.at(d) - gamma_lower_local.at(d));
          double sol_to_ub_sigma2 = std::abs(gamma_tmp.at(d) - gamma_upper_local.at(d));
          double sol_to_b_sigma2 = std::min(sol_to_ub_sigma2, sol_to_lb_sigma2);
          double sol_to_b = sol_to_b_theta < sol_to_b_sigma2 ? sol_to_b_theta : sol_to_b_sigma2;
          
          if ((retry < Optim::max_restart)
              && ((opt_result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0)
                  || ((sol_to_b < arma::datum::eps) && (opt_result.num_iters <= 2))
                  || (opt_result.f_opt > best_f_opt))) {
            gamma_tmp.head(d)
                = (theta0.row(start_idx % multistart).t() + theta_lower)
                  / pow(2.0, retry + 1);
            gamma_tmp.at(d) = sigma20[start_idx % sigma20.n_elem];

            if (Optim::reparametrize)
              gamma_tmp = Optim::reparam_to(gamma_tmp);

            gamma_lower_local = arma::min(gamma_tmp, gamma_lower_local);
            gamma_upper_local = arma::max(gamma_tmp, gamma_upper_local);

            retry++;
          } else {
            break;
          }
        }

        // Final evaluation to update model
        double min_ofn_tmp = fit_ofn(best_gamma, nullptr, &m);

        result.objective_value = min_ofn_tmp;
        result.gamma = best_gamma;
        result.theta_sigma2 = Optim::reparametrize ? Optim::reparam_from(best_gamma) : best_gamma;

        // Copy (not move) since m is a reference to preallocated memory
        // Force DEEP copy to avoid any shared memory issues
        result.L = arma::mat(m.L);  // Force copy constructor
        result.R = arma::mat(m.R);  // Force copy constructor
        result.Fstar = arma::mat(m.Fstar);  // Force copy constructor
        result.Rstar = arma::mat(m.Rstar);  // Force copy constructor
        result.Qstar = arma::mat(m.Qstar);  // Force copy constructor
        result.Estar = arma::vec(m.Estar);  // Force copy constructor
        result.ystar = arma::vec(m.ystar);  // Force copy constructor
        result.SSEstar = m.SSEstar;
        result.betahat = arma::vec(m.betahat);  // Force copy constructor
        result.success = true;

      } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        if (Optim::log_level > 0) {
          arma::cout << "Warning: start point " << (start_idx + 1) << " failed: " << e.what() << arma::endl;
        }
      }

      return result;
    };

    // Execute optimizations (parallel if multistart > 1)
    std::vector<OptimizationResult> results(multistart);
    std::mutex results_mutex;  // Protect results vector writes

    if (multistart == 1) {
      results[0] = optimize_worker(0);
    } else {
      // Determine thread pool size
      unsigned int n_cpu = std::thread::hardware_concurrency();
      int pool_size = Optim::thread_pool_size;
      if (pool_size <= 0) {
        pool_size = std::max(1u, n_cpu);
      }
      pool_size = std::min(pool_size, (int)multistart);  // Don't exceed number of tasks
      
      if (Optim::log_level > 0) {
        arma::cout << "Thread pool: " << pool_size << " workers (ncpu=" << n_cpu 
                   << ", multistart=" << multistart << ")" << arma::endl;
      }
      
      // Thread pool implementation: use semaphore-like counter
      std::atomic<int> next_task(0);
      std::vector<std::thread> threads;
      threads.reserve(pool_size);
      
      for (int worker_id = 0; worker_id < pool_size; worker_id++) {
        threads.emplace_back([&, worker_id]() {
          while (true) {
            int task_id = next_task.fetch_add(1);
            if (task_id >= multistart) break;
            
            // Add staggered startup delay to avoid thread initialization race conditions
            int delay_ms = task_id * Optim::thread_start_delay_ms;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            
            OptimizationResult local_result = optimize_worker(task_id);
            
            {
              std::lock_guard<std::mutex> lock(results_mutex);
              // Deep copy each matrix to ensure no shared memory
              results[task_id].start_index = local_result.start_index;
              results[task_id].objective_value = local_result.objective_value;
              results[task_id].gamma = arma::vec(local_result.gamma);
              results[task_id].theta_sigma2 = arma::vec(local_result.theta_sigma2);
              results[task_id].L = arma::mat(local_result.L);
              results[task_id].R = arma::mat(local_result.R);
              results[task_id].Fstar = arma::mat(local_result.Fstar);
              results[task_id].Rstar = arma::mat(local_result.Rstar);
              results[task_id].Qstar = arma::mat(local_result.Qstar);
              results[task_id].Estar = arma::vec(local_result.Estar);
              results[task_id].ystar = arma::vec(local_result.ystar);
              results[task_id].SSEstar = local_result.SSEstar;
              results[task_id].betahat = arma::vec(local_result.betahat);
              results[task_id].success = local_result.success;
              results[task_id].error_message = local_result.error_message;
            }
          }
        });
      }
      
      for (auto& t : threads) {
        if (t.joinable()) {
          t.join();
        }
      }
    }

    // Find best result
    int best_idx = -1;
    int successful_optimizations = 0;

    for (size_t i = 0; i < results.size(); i++) {
      const auto& r = results[i];
      if (r.success) {
        successful_optimizations++;
        if (r.objective_value < min_ofn) {
          min_ofn = r.objective_value;
          best_idx = static_cast<int>(i);
        }
      }
    }

    if (successful_optimizations == 0) {
      throw std::runtime_error("All " + std::to_string(multistart)
                               + " optimization attempts failed");
    }

    if (Optim::log_level > 0 && successful_optimizations < multistart) {
      arma::cout << "\nOptimization summary: " << successful_optimizations << "/" << multistart
                << " succeeded" << arma::endl;
    }

    // Update member variables with best result
    if (best_idx >= 0) {
      const auto& best = results[best_idx];
      m_theta = best.theta_sigma2.head(d);  // copy
      m_est_theta = true;
      m_is_empty = false;
      m_T = best.L;          // copy instead of move to avoid issues
      m_R = best.R;
      m_M = best.Fstar;
      m_circ = best.Rstar;
      m_star = best.Qstar;

      if (m_est_beta) {
        m_beta = best.betahat;
        m_z = best.Estar;
      } else {
        m_z = best.ystar - m_M * m_beta;
      }

      if (m_est_sigma2) {
        m_sigma2 = best.theta_sigma2.at(d);
      }

      if (Optim::log_level > 0) {
        arma::cout << "\nBest solution from start point " << (best_idx + 1) << " with objective: " << min_ofn
                  << arma::endl;
      }
    }
  } else
    throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#])");

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param X_n is n_n*d matrix of points where to predict output
 * @param return_stdev is true if return also stdev column vector
 * @param return_cov is true if return also cov matrix between X_n
 * @param return_deriv is true if return also derivative of prediction wrt x
 * @return output prediction: n_n means, [n_n standard deviations], [n_n*n_n full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat>
NoiseKriging::predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::vec yhat_n(n_n);
  arma::vec ysd2_n = arma::vec(n_n, arma::fill::zeros);
  arma::mat Sigma_n = arma::mat(n_n, n_n, arma::fill::zeros);
  arma::mat Dyhat_n = arma::mat(n_n, d, arma::fill::zeros);
  arma::mat Dysd2_n = arma::mat(n_n, d, arma::fill::zeros);

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  Xn_n = trans(Xn_n);

  auto t0 = Bench::tic();
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  for (arma::uword i = 0; i < n_o; i++) {
    for (arma::uword j = 0; j < n_n; j++) {
      R_on.at(i, j) = _Cov((Xn_o.col(i) - Xn_n.col(j)), m_theta);
    }
  }
  R_on *= m_sigma2;
  t0 = Bench::toc(nullptr, "R_on       ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n     ", t0);

  // Un-normalize predictor
  yhat_n = m_centerY + m_scaleY * yhat_n;

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n    ", t0);

  if (return_stdev) {
    ysd2_n = m_sigma2 - sum(Rstar_on % Rstar_on, 0).as_col() + sum(Ecirc_n % Ecirc_n, 1).as_col();
    ysd2_n.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
    ysd2_n *= m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "ysd2_n     ", t0);
  }

  if (return_cov) {
    // Compute the covariance matrix between new data points
    arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, m_sigma2);
    t0 = Bench::toc(nullptr, "R_nn       ", t0);

    Sigma_n = R_nn - trans(Rstar_on) * Rstar_on + Ecirc_n * trans(Ecirc_n);
    Sigma_n *= m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "Sigma_n    ", t0);
  }

  if (return_deriv) {
    //// https://github.com/libKriging/dolka/blob/bb1dbf0656117756165bdcff0bf5e0a1f963fbef/R/kmStuff.R#L322C1-L363C10
    // for (i in 1:n_n) {
    //
    //   ## =================================================================
    //   ## 'DF_n_i' and 'DR_on_i' are matrices with
    //   ## dimension c(n_n, d)
    //   ## =================================================================
    //
    //   DF_n_i <- trend.deltax(x = XNew[i, ], model = object)
    //   KOldNewi <- as.vector(KOldNew[ , i])
    //   DR_on_i <- covVector.dx(x = as.vector(XNew[i, ]),
    //                               X = X,
    //                               object = object@covariance,
    //                               c = KOldNewi)
    //
    //   KOldNewStarDer[ , i, i, ] <-
    //       backsolve(L, DR_on_i, upper.tri = FALSE)
    //
    //   ## Gradient of the kriging trend and mean
    //   muNewHatDer[i, i, ] <- crossprod(DF_n_i, betaHat)
    //   ## dim in product c(d, n) and NULL(length d)
    //   mean_nHatDer[i, i, ] <- muNewHatDer[i, i, ] +
    //       crossprod(KOldNewStarDer[ , i, i,  ], zStar)
    //   ## dim in product c(d, n) and NULL(length n)
    //   s2Der[i, i,  ] <-
    //       - 2 * crossprod(KOldNewStarDer[ , i, i, ],
    //                       drop(KOldNewStar[ , i]))
    //
    //   ## dim in product c(d, n) and c(n, p)
    //
    //   if (type == "UK") {
    //       ENewDagDer[i, i, , ] <-
    //           backsolve(t(RStar),
    //                     DF_n_i -
    //                     t(crossprod(KOldNewStarDer[ , i, i, ], FStar)),
    //                     upper.tri = FALSE)
    //       ## dim in product NULL (length p) and c(p, d) because of 'drop'
    //       s2Der[i, i, ] <- s2Der[i, i, ] + 2 * drop(ENewDagT[ , i]) %*%
    //           drop(ENewDagDer[i, i, , ])
    //   }
    //  numerical derivative step : value is sensitive only for non linear trends. Otherwise, it gives exact results.
    const double h = 1.0E-5;
    arma::mat h_eye_d = h * arma::mat(d, d, arma::fill::eye);

    // Compute the derivatives of the covariance and trend functions
    for (arma::uword i = 0; i < n_n; i++) {  // for each predict point... should be parallel ?
      arma::mat DR_on_i = arma::mat(n_o, d, arma::fill::none);
      for (arma::uword j = 0; j < n_o; j++) {
        DR_on_i.row(j) = R_on.at(j, i) * trans(_DlnCovDx(Xn_n.col(i) - Xn_o.col(j), m_theta));
      }
      t0 = Bench::toc(nullptr, "DR_on_i    ", t0);

      arma::mat tXn_n_repd_i
          = arma::trans(Xn_n.col(i) * arma::mat(1, d, arma::fill::ones));  // just duplicate X_n.row(i) d times
      arma::mat DF_n_i = (Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i + h_eye_d)
                          - Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i - h_eye_d))
                         / (2 * h);
      t0 = Bench::toc(nullptr, "DF_n_i     ", t0);

      arma::mat W_i = LinearAlgebra::solve(m_T, DR_on_i);
      t0 = Bench::toc(nullptr, "W_i        ", t0);
      Dyhat_n.row(i) = trans(DF_n_i * m_beta + trans(W_i) * m_z);
      t0 = Bench::toc(nullptr, "Dyhat_n    ", t0);

      if (return_stdev) {
        arma::mat DEcirc_n_i = LinearAlgebra::solve(m_circ.t(), trans(DF_n_i - W_i.t() * m_M));
        Dysd2_n.row(i) = -2 * Rstar_on.col(i).t() * W_i + 2 * Ecirc_n.row(i) * DEcirc_n_i;
        t0 = Bench::toc(nullptr, "Dysd2_n    ", t0);
      }
    }
    Dyhat_n *= m_scaleY;
    Dysd2_n *= m_scaleY * m_scaleY;
  }

  return std::make_tuple(std::move(yhat_n),
                         std::move(arma::sqrt(ysd2_n)),
                         std::move(Sigma_n),
                         std::move(Dyhat_n),
                         std::move(Dysd2_n / (2 * arma::sqrt(ysd2_n) * arma::mat(1, d, arma::fill::ones))));
  /*if (return_stdev)
    if (return_cov)
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), std::move(pred_cov));
    else
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), nullptr);
  else if (return_cov)
    return std::make_tuple(std::move(yhat_n), std::move(pred_cov), nullptr);
  else
    return std::make_tuple(std::move(yhat_n), nullptr, nullptr);*/
}

/** Draw sample trajectories of kriging at given points X'
 * @param X_n is n_n*d matrix of points where to simulate output
 * @param seed is seed for random number generator
 * @param nsim is number of simulations to draw
 * @param with_noise is n_n (or 1) vector of noise to add to simulations
 * @param will_update is true if we want to keep simulations data for future update
 * @return output is n_n*nsim matrix of simulations at X_n
 */
LIBKRIGING_EXPORT arma::mat NoiseKriging::simulate(const int nsim,
                                                   const int seed,
                                                   const arma::mat& X_n,
                                                   const arma::vec& with_noise,
                                                   const bool will_update) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  if (with_noise.n_elem > 1 && with_noise.n_elem != n_n)
    throw std::runtime_error("Noise vector should have same length as X_n: " + std::to_string(with_noise.n_elem)
                             + " instead of " + std::to_string(n_n) + " (or 0 if no noise)");

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  Xn_n = trans(Xn_n);

  auto t0 = Bench::tic();
  // Compute covariance between new data
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, m_sigma2);
  t0 = Bench::toc(nullptr, "R_nn          ", t0);

  // Compute covariance between training data and new data to predict
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, Xn_o, Xn_n, m_theta, _Cov, m_sigma2);
  t0 = Bench::toc(nullptr, "R_on        ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  arma::vec yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n        ", t0);

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n       ", t0);

  arma::mat SigmaNoTrend_nKo = R_nn - trans(Rstar_on) * Rstar_on;
  arma::mat Sigma_nKo = SigmaNoTrend_nKo + Ecirc_n * trans(Ecirc_n);
  t0 = Bench::toc(nullptr, "Sigma_nKo     ", t0);

  arma::mat LSigma_nKo = LinearAlgebra::safe_chol_lower(
      Sigma_nKo / m_sigma2);  // normalization to keep same diag inc (+=1e-10) than other Kriging
  t0 = Bench::toc(nullptr, "LSigma_nKo     ", t0);

  arma::mat y_n = arma::mat(n_n, nsim, arma::fill::none);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma_nKo * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  // Un-normalize simulations
  y_n = m_centerY + m_scaleY * y_n;

  if (will_update) {
    lastsimup_Xn_u.clear();  // force reset to force update_simulate consider new data
    lastsim_y_n = y_n;
    lastsim_with_noise = with_noise;

    lastsim_Xn_n = Xn_n;
    lastsim_seed = seed;
    lastsim_nsim = nsim;

    lastsim_R_nn = R_nn;
    lastsim_F_n = F_n;

    lastsim_L_oCn = Rstar_on;
    lastsim_L_nCn = LinearAlgebra::safe_chol_lower(SigmaNoTrend_nKo);
    t0 = Bench::toc(nullptr, "L_nCn     ", t0);

    lastsim_L_on = arma::join_rows(arma::join_cols(m_T, lastsim_L_oCn.t()),
                                   arma::join_cols(arma::zeros(n_o, n_n), lastsim_L_nCn));

    arma::mat Linv_on = LinearAlgebra::solve(lastsim_L_on, arma::mat(n_o + n_n, n_o + n_n, arma::fill::eye));
    t0 = Bench::toc(nullptr, "Linv_on     ", t0);
    lastsim_Rinv_on = Linv_on.t() * Linv_on;
    t0 = Bench::toc(nullptr, "Rinv_on     ", t0);

    lastsim_F_on = arma::join_cols(m_F, lastsim_F_n);
    lastsim_Fstar_on = LinearAlgebra::solve(lastsim_L_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fstar_on     ", t0);
    arma::mat Q_Fstar_on;
    arma::qr(Q_Fstar_on, lastsim_circ_on, lastsim_Fstar_on);
    lastsim_Fcirc_on = LinearAlgebra::rsolve(lastsim_circ_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fcirc_on     ", t0);

    lastsim_Fhat_nKo = lastsim_L_oCn.t() * m_M;
    t0 = Bench::toc(nullptr, "Fhat_nKo     ", t0);
    lastsim_Ecirc_nKo = LinearAlgebra::rsolve(m_circ, F_n - lastsim_Fhat_nKo);
    t0 = Bench::toc(nullptr, "Ecirc_nKo     ", t0);
  }

  // Add noise
  arma::mat eps = arma::mat(n_n, nsim, arma::fill::none);
  if (with_noise.n_elem == 1)
    eps = with_noise.at(0) * Random::randn_mat(n_n, nsim);
  else if (with_noise.n_elem == n_n) {
    eps.each_col() = with_noise;
    eps = eps % Random::randn_mat(n_n, nsim);
  }

  return y_n + eps;
}

LIBKRIGING_EXPORT arma::mat NoiseKriging::update_simulate(const arma::vec& y_u,
                                                          const arma::vec& noise_u,
                                                          const arma::mat& X_u) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  if (noise_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Noise vector should have same length as X_u: " + std::to_string(noise_u.n_elem)
                             + " instead of " + std::to_string(X_u.n_rows));

  if (lastsim_y_n.is_empty() || lastsim_y_n.n_rows == 0)
    throw std::runtime_error("No previous simulation data available");

  if (lastsim_Xn_n.n_rows != X_u.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(X_u.n_cols)
                             + "), last sim X: (...x" + std::to_string(lastsim_Xn_n.n_rows) + ")");

  arma::uword n_n = lastsim_Xn_n.n_cols;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::mat Xn_o = trans(m_X);    // already normalized if needed
  arma::mat Xn_n = lastsim_Xn_n;  // already normalized

  arma::uword n_on = n_o + n_n;
  // arma::mat Xn_on = arma::join_cols(Xn_o, Xn_n);
  arma::mat F_on = arma::join_cols(m_F, lastsim_F_n);

  arma::uword n_u = X_u.n_rows;
  // Normalize X_u
  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_u = Trend::regressionModelMatrix(m_regmodel, Xn_u);

  auto t0 = Bench::tic();
  Xn_u = trans(Xn_u);
  t0 = Bench::toc(nullptr, "Xn_u.t()      ", t0);

  bool use_lastsimup = (!lastsimup_Xn_u.is_empty()) && (lastsimup_Xn_u - Xn_u).is_zero(arma::datum::eps)
                       && (!lastsimup_noise_u.is_empty()) && (lastsimup_noise_u - noise_u).is_zero(arma::datum::eps);
  if (!use_lastsimup) {
    lastsimup_Xn_u = Xn_u;

    // Compute covariance between updated data
    lastsimup_R_uu = arma::mat(n_u, n_u, arma::fill::none);
    arma::vec diag_uu = m_sigma2 + noise_u;
    LinearAlgebra::covMat_sym_X(&lastsimup_R_uu, Xn_u, m_theta, _Cov, m_sigma2, diag_uu);
    t0 = Bench::toc(nullptr, "R_uu          ", t0);

    // Compute covariance between updated/old data
    lastsimup_R_uo = arma::mat(n_u, n_o, arma::fill::none);
    LinearAlgebra::covMat_rect(&lastsimup_R_uo, Xn_u, Xn_o, m_theta, _Cov, m_sigma2);
    t0 = Bench::toc(nullptr, "R_uo          ", t0);

    // Compute covariance between updated/new data
    lastsimup_R_un = arma::mat(n_u, n_n, arma::fill::none);
    LinearAlgebra::covMat_rect(&lastsimup_R_un, Xn_u, Xn_n, m_theta, _Cov, m_sigma2);
    t0 = Bench::toc(nullptr, "R_un          ", t0);
  }

  // ======================================================================
  // FOXY step #1 Extend the simulation to the design 'X_u' IF
  // NECESSARY. Remind that the simulation number j is
  // conditional on the given 'y_o' and on 'y_n = Y_sim[ , j]'
  //
  // CAUTION. To avoid unnecessary re-computations we here use
  // auxiliary variables that where computed in the creation of
  // the KM0 step AND later in the simulation. The first ones are
  // in 'theKM0$Extra' and the second ones are in 'Ex'
  //
  // In indices 'C' means comma and 'K' means pipe '|'
  //
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat R_onCu = arma::join_rows(lastsimup_R_uo, lastsimup_R_un).t();
    arma::mat Rstar_onCu = LinearAlgebra::solve(lastsim_L_on, R_onCu);
    t0 = Bench::toc(nullptr, "Rstar_onCu          ", t0);

    arma::mat Ecirc_uKon
        = LinearAlgebra::rsolve(lastsim_circ_on, F_u - Rstar_onCu.t() * lastsim_Fstar_on) * std::sqrt(m_sigma2);
    t0 = Bench::toc(nullptr, "Ecirc_uKon          ", t0);

    arma::mat Sigma_uKon = lastsimup_R_uu - Rstar_onCu.t() * Rstar_onCu + Ecirc_uKon * Ecirc_uKon.t();
    t0 = Bench::toc(nullptr, "Sigma_uKon          ", t0);

    arma::mat LSigma_uKon = LinearAlgebra::safe_chol_lower(
        Sigma_uKon / m_sigma2);  // normalization to keep same diag inc (+=1e-10) than other Kriging
    t0 = Bench::toc(nullptr, "LSigma_uKon          ", t0);

    arma::mat W_uCon = (R_onCu.t() + Ecirc_uKon * lastsim_Fcirc_on.t()) * lastsim_Rinv_on;
    t0 = Bench::toc(nullptr, "W_uCon          ", t0);

    arma::mat m_u = W_uCon.head_cols(n_o) * m_y;
    arma::mat M_u = arma::repmat(m_u, 1, lastsim_nsim) + W_uCon.tail_cols(n_n) * lastsim_y_n;

    Random::reset_seed(lastsim_seed);
    lastsimup_y_u = M_u + LSigma_uKon * Random::randn_mat(n_u, lastsim_nsim) * std::sqrt(m_sigma2);
    t0 = Bench::toc(nullptr, "y_u          ", t0);
  }

  // ======================================================================
  // FOXY step #2   Update the simulated paths on 'X_n'
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat Rstar_ou = LinearAlgebra::solve(m_T, lastsimup_R_uo.t());
    t0 = Bench::toc(nullptr, "Rstar_ou          ", t0);

    arma::mat Fhat_uKo = Rstar_ou.t() * m_M;
    arma::mat Ecirc_uKo = LinearAlgebra::rsolve(m_circ, F_u - Fhat_uKo);
    t0 = Bench::toc(nullptr, "Ecirc_uKo          ", t0);

    arma::mat Rtild_uCu = lastsimup_R_uu - Rstar_ou.t() * Rstar_ou + Ecirc_uKo * Ecirc_uKo.t();
    t0 = Bench::toc(nullptr, "Rtild_uCu          ", t0);

    arma::mat Rtild_nCu = lastsimup_R_un - Rstar_ou.t() * lastsim_L_oCn + Ecirc_uKo * lastsim_Ecirc_nKo.t();
    t0 = Bench::toc(nullptr, "Rtild_nCu          ", t0);

    lastsimup_Wtild_nKu = LinearAlgebra::solve(Rtild_uCu, Rtild_nCu).t();
    t0 = Bench::toc(nullptr, "Wtild_nKu          ", t0);
  }

  // Add noise
  arma::mat eps = arma::mat(n_n, lastsim_nsim, arma::fill::none);
  if (lastsim_with_noise.n_elem == 1)
    eps = lastsim_with_noise.at(0) * Random::randn_mat(n_n, lastsim_nsim);
  else if (lastsim_with_noise.n_elem == n_n) {
    eps.each_col() = lastsim_with_noise;
    eps = eps % Random::randn_mat(n_n, lastsim_nsim);
  }

  return lastsim_y_n + lastsimup_Wtild_nKu * (arma::repmat(y_u, 1, lastsim_nsim) - lastsimup_y_u) + eps;
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param noise_u is n_u length column vector of new output variance
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void NoiseKriging::update(const arma::vec& y_u,
                                            const arma::vec& noise_u,
                                            const arma::mat& X_u,
                                            const bool refit) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (noise_u.n_elem != y_u.n_elem)
    throw std::runtime_error("Dimension of new data should be the same:\n noise: (" + std::to_string(noise_u.n_elem)
                             + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  // rebuild starting parameters
  Parameters parameters;
  if (refit) {  // re-fit
    if (m_est_beta)
      parameters
          = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY * arma::ones<arma::vec>(1)),
                       this->m_est_sigma2,
                       std::make_optional(trans(this->m_theta) % this->m_scaleX),
                       this->m_est_theta,
                       std::make_optional(arma::ones<arma::vec>(0)),
                       true};
    else
      parameters
          = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY * arma::ones<arma::vec>(1)),
                       this->m_est_sigma2,
                       std::make_optional(trans(this->m_theta) % this->m_scaleX),
                       this->m_est_theta,
                       std::make_optional(trans(this->m_beta) * this->m_scaleY),
                       false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                              y_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols(m_noise * this->m_scaleY * this->m_scaleY,
                              noise_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
              m_regmodel,
              m_normalize,
              m_optim,
              m_objective,
              parameters);
  } else {  // just update
    parameters
        = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY * arma::ones<arma::vec>(1)),
                     false,
                     std::make_optional(trans(arma::mat(this->m_theta)) % this->m_scaleX),
                     false,
                     std::make_optional(arma::vec(this->m_beta) * this->m_scaleY),
                     false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                              y_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols(m_noise * this->m_scaleY * this->m_scaleY,
                              noise_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
              m_regmodel,
              m_normalize,
              "none",
              m_objective,
              parameters);
  }
}

LIBKRIGING_EXPORT std::string NoiseKriging::summary() const {
  std::ostringstream oss;
  auto vec_printer = [&oss](const arma::vec& v) {
    v.for_each([&oss, i = 0](const arma::vec::elem_type& val) mutable {
      if (i++ > 0)
        oss << ", ";
      oss << val;
    });
  };

  if (m_X.is_empty() || m_X.n_rows == 0) {  // not yet fitted
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
  } else {
    oss << "* data";
    oss << ((m_normalize) ? " (normalized): " : ": ") << m_X.n_rows << "x";
    arma::rowvec Xmins = arma::min(m_X, 0);
    arma::rowvec Xmaxs = arma::max(m_X, 0);
    for (arma::uword i = 0; i < m_X.n_cols; i++) {
      oss << "[" << Xmins[i] << "," << Xmaxs[i] << "]";
      if (i < m_X.n_cols - 1)
        oss << ",";
    }
    oss << " -> " << m_y.n_elem << "x[" << arma::min(m_y) << "," << arma::max(m_y) << "]\n";
    oss << "  * noise: " << m_noise.n_elem << "x[" << arma::min(m_noise) << "," << arma::max(m_noise) << "]\n";
    oss << "* trend " << Trend::toString(m_regmodel);
    oss << ((m_est_beta) ? " (est.): " : ": ");
    vec_printer(m_beta);
    oss << "\n";
    oss << "* variance";
    oss << ((m_est_sigma2) ? " (est.): " : ": ");
    oss << m_sigma2;
    oss << "\n";
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
    oss << "  * range";
    oss << ((m_est_theta) ? " (est.): " : ": ");
    vec_printer(m_theta);
    oss << "\n";
    oss << "  * fit:\n";
    oss << "    * objective: " << m_objective << "\n";
    oss << "    * optim: " << m_optim << "\n";
  }
  return oss.str();
}

void NoiseKriging::save(const std::string filename) const {
  nlohmann::json j;

  j["version"] = 2;
  j["content"] = "NoiseKriging";

  // _Cov_pow & std::function embedded by make_Cov
  j["covType"] = m_covType;
  j["X"] = to_json(m_X);
  j["centerX"] = to_json(m_centerX);
  j["scaleX"] = to_json(m_scaleX);
  j["y"] = to_json(m_y);
  j["centerY"] = m_centerY;
  j["scaleY"] = m_scaleY;
  j["normalize"] = m_normalize;
  j["noise"] = to_json(m_noise);
  j["regmodel"] = Trend::toString(m_regmodel);
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  // Auxiliary data
  j["dX"] = to_json(m_dX);
  j["maxdX"] = to_json(m_maxdX);
  j["F"] = to_json(m_F);
  j["T"] = to_json(m_T);
  j["R"] = to_json(m_R);
  j["M"] = to_json(m_M);
  j["star"] = to_json(m_star);
  j["circ"] = to_json(m_circ);
  j["z"] = to_json(m_z);
  j["beta"] = to_json(m_beta);
  j["est_beta"] = m_est_beta;
  j["theta"] = to_json(m_theta);
  j["est_theta"] = m_est_theta;
  j["sigma2"] = m_sigma2;
  j["est_sigma2"] = m_est_sigma2;

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

NoiseKriging NoiseKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  }
  std::string content = j["content"].template get<std::string>();
  if (content != "NoiseKriging") {
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'NoiseKriging'"));
  }

  std::string covType = j["covType"].template get<std::string>();
  NoiseKriging kr(covType);  // _Cov_pow & std::function embedded by make_Cov

  kr.m_X = mat_from_json(j["X"]);
  kr.m_centerX = rowvec_from_json(j["centerX"]);
  kr.m_scaleX = rowvec_from_json(j["scaleX"]);
  kr.m_y = colvec_from_json(j["y"]);
  kr.m_centerY = j["centerY"].template get<decltype(kr.m_centerY)>();
  kr.m_scaleY = j["scaleY"].template get<decltype(kr.m_scaleY)>();
  kr.m_normalize = j["normalize"].template get<decltype(kr.m_normalize)>();
  kr.m_noise = colvec_from_json(j["noise"]);

  std::string model = j["regmodel"].template get<std::string>();
  kr.m_regmodel = Trend::fromString(model);

  kr.m_optim = j["optim"].template get<decltype(kr.m_optim)>();
  kr.m_objective = j["objective"].template get<decltype(kr.m_objective)>();
  // Auxiliary data
  kr.m_dX = mat_from_json(j["dX"]);
  kr.m_maxdX = colvec_from_json(j["maxdX"]);
  kr.m_F = mat_from_json(j["F"]);
  kr.m_T = mat_from_json(j["T"]);
  kr.m_R = mat_from_json(j["R"]);
  kr.m_M = mat_from_json(j["M"]);
  kr.m_star = mat_from_json(j["star"]);
  kr.m_circ = mat_from_json(j["circ"]);
  kr.m_z = colvec_from_json(j["z"]);
  kr.m_beta = colvec_from_json(j["beta"]);
  kr.m_est_beta = j["est_beta"].template get<decltype(kr.m_est_beta)>();
  kr.m_theta = colvec_from_json(j["theta"]);
  kr.m_est_theta = j["est_theta"].template get<decltype(kr.m_est_theta)>();
  kr.m_sigma2 = j["sigma2"].template get<decltype(kr.m_sigma2)>();
  kr.m_est_sigma2 = j["est_sigma2"].template get<decltype(kr.m_est_sigma2)>();
  kr.m_is_empty = false;

  return kr;
}
