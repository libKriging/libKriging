#include "libKriging/LinearRegressionOptim.hpp"

// @ref: https://www.kthohr.com/optimlib.html
#include <optim.hpp>

LIBKRIGING_EXPORT
LinearRegressionOptim::LinearRegressionOptim() : m_sig2{} {};

struct err_fn_data {
  arma::vec y;
  arma::mat X;
};

double err_fn(const arma::vec& coef, arma::vec* grad_out, err_fn_data* fn_data) {
  err_fn_data* d = fn_data;

  arma::vec y_est = d->X * coef;

  std::cout << "****************" << std::endl;
  arma::cout << "coef: " << coef << arma::endl;

  double err = arma::sum(arma::square(y_est - d->y));
  std::cout << "Err: " << err << std::endl;

  if (grad_out != nullptr) {
    int k = coef.n_elem;
    for (int i = 0; i < k; i++) {
      arma::vec coef2 = coef;
      coef2(i) = coef2(i) + 0.0001;

      arma::vec y_est2 = d->X * coef2;
      double err2 = arma::sum(arma::square(y_est2 - d->y));

      (*grad_out)(i) = (err2 - err) / 0.0001;
    }
    arma::cout << "Grad: " << *grad_out << arma::endl;
  }
  // arma::cout<<"Y = X * s :\n"<<y_est<<arma::endl;

  return err;
}

LIBKRIGING_EXPORT
// returned object should hold error state instead of void
void LinearRegressionOptim::fit(const arma::vec& y, const arma::mat& X) {
  int n = X.n_rows;
  int k = X.n_cols;

  // We will replace that by a BFGS optimization. Just as a proof of concept for BFGS usage.
  // coef = arma::solve(X, y);
  m_coef = arma::ones(k);
  arma::cout << "Initial solution vector :\n" << m_coef << arma::endl;
  optim::algo_settings_t algo_settings;
  err_fn_data fn_data{y, X};
  algo_settings.iter_max = 100;
  algo_settings.grad_err_tol = 0.01;
  algo_settings.print_level = 4;
  algo_settings.conv_failure_switch = 2;
  bool bfgs_ok = optim::bfgs(
      m_coef,
      [&fn_data](const arma::vec& vals_inp, arma::vec* grad_out, void*) -> double {
        return err_fn(vals_inp, grad_out, &fn_data);
      },
      nullptr,
      algo_settings);
  if (!bfgs_ok) {
    std::cout.flush();
    throw std::runtime_error("BFGS failed");
  }
  arma::cout << "Coef: " << m_coef << arma::endl;
  arma::colvec resid = y - X * m_coef;

  m_sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  m_stderrest = arma::sqrt(m_sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));
}

std::tuple<arma::colvec, arma::colvec> LinearRegressionOptim::predict(const arma::mat& X) {
  // should test that X.n_cols == fit.X.n_cols
  // int n = X.n_rows;
  // int k = X.n_cols;

  arma::colvec y = X * m_coef;
  arma::colvec stderr_v = arma::sqrt(arma::diagvec(X * arma::diagmat(m_stderrest) * arma::trans(X)));

  return std::make_tuple(std::move(y), std::move(stderr_v));
}