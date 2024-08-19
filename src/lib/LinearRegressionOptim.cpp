#include "libKriging/LinearRegressionOptim.hpp"

#include "libKriging/utils/data_from_arma_vec.hpp"

#include <lbfgsb_cpp/lbfgsb.hpp>
#include "libKriging/Optim.hpp"

LIBKRIGING_EXPORT
LinearRegressionOptim::LinearRegressionOptim() : m_sig2{} {};

struct err_fn_data {
  arma::vec y;
  arma::mat X;
};

double err_fn(const arma::vec &coef, arma::vec *grad_out, err_fn_data *fn_data) {
  err_fn_data *d = fn_data;

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
void LinearRegressionOptim::fit(const arma::vec &y, const arma::mat &X) {
  arma::uword n = X.n_rows;
  arma::uword k = X.n_cols;

  // We will replace that by a BFGS optimization. Just as a proof of concept for BFGS usage.
  // coef = arma::solve(X, y);
  m_coef = arma::ones(k);
  arma::cout << "Initial solution vector :\n" << m_coef << arma::endl;

  // like in Kriging.cpp
  arma::vec theta_lower = Optim::theta_lower_factor * trans(max(X, 0) - min(X, 0));
  arma::vec theta_upper = Optim::theta_upper_factor * trans(max(X, 0) - min(X, 0));
  arma::vec gamma_lower = theta_lower;
  arma::vec gamma_upper = theta_upper;

  lbfgsb::Optimizer optimizer{k};
  optimizer.iprint = -1;
  optimizer.max_iter = 100;
  optimizer.pgtol = 0.01;
  optimizer.factr = 1e7;
  arma::ivec bounds_type{k, arma::fill::value(2)};  // means both upper & lower bounds

  err_fn_data fn_data{y, X};
  auto result
      = optimizer.minimize([&fn_data](const arma::vec &vals_inp,
                                      arma::vec &grad_out) -> double { return err_fn(vals_inp, &grad_out, &fn_data); },
                           m_coef,
                           gamma_lower.memptr(),
                           gamma_upper.memptr(),
                           bounds_type.memptr());
  arma::cout << "Coef: " << m_coef << arma::endl;
  arma::colvec resid = y - X * m_coef;

  m_sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  m_stderrest = arma::sqrt(m_sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));
}

std::tuple<arma::colvec, arma::colvec> LinearRegressionOptim::predict(const arma::mat &X) {
  // should test that X.n_cols == fit.X.n_cols
  // int n = X.n_rows;
  // int k = X.n_cols;

  arma::colvec y = X * m_coef;
  arma::colvec stderr_v = arma::sqrt(arma::diagvec(X * arma::diagmat(m_stderrest) * arma::trans(X)));

  return std::make_tuple(std::move(y), std::move(stderr_v));
}