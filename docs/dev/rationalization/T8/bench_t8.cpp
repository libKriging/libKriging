// T8 bench: cost and quality of the MLPKriging variance calibration.
// Usage: bench_t8 <out_file> <folds> [optim] [dataset filter] [reps]
// Env (experimental hooks): LK_T8_START=fit, LK_T8_ITER=<n>, LK_T8_DUMP=<file>
#define _USE_MATH_DEFINES
#include <cmath>
#include "libKriging/MLPKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace libKriging;

static double cpu_s() {
  rusage r;
  getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec + r.ru_stime.tv_sec + 1e-6 * r.ru_stime.tv_usec;
}

static arma::mat lhs(arma::uword n, arma::uword d) {
  arma::mat X(n, d);
  for (arma::uword j = 0; j < d; ++j)
    X.col(j) = (arma::conv_to<arma::vec>::from(arma::randperm(n)) + arma::randu<arma::vec>(n)) / n;
  return X;
}

using Fn = std::function<arma::vec(const arma::mat&)>;
struct Case {
  std::string name;
  arma::uword d, n;
  Fn f;
};

static double branin1(double x1, double x2) {
  const double pi = M_PI;
  return std::pow(x2 - 5.1 / (4 * pi * pi) * x1 * x1 + 5 / pi * x1 - 6, 2) + 10 * (1 - 1 / (8 * pi)) * std::cos(x1)
         + 10;
}

static arma::vec rowmap(const arma::mat& X, std::function<double(const arma::rowvec&)> g) {
  arma::vec y(X.n_rows);
  for (arma::uword i = 0; i < X.n_rows; ++i)
    y(i) = g(X.row(i));
  return y;
}

static arma::vec branin_proj5(const arma::mat& X) {
  arma::mat W = {{0.9, -0.4, 0.7, 0.2, -0.5}, {-0.3, 0.8, 0.1, -0.6, 0.4}};
  W.each_col() /= arma::sum(arma::abs(W), 1);
  arma::mat U = X * W.t();
  U.each_row() -= arma::sum(arma::clamp(W, -arma::datum::inf, 0.0), 1).t();
  arma::vec y(X.n_rows);
  for (arma::uword i = 0; i < X.n_rows; ++i)
    y(i) = branin1(15 * U(i, 0) - 5, 15 * U(i, 1));
  return y;
}

static arma::vec branin_proj10(const arma::mat& X) {
  arma::mat W = {{0.9, -0.4, 0.7, 0.2, -0.5, 0.3, -0.1, 0.6, 0.0, -0.2},
                 {-0.3, 0.8, 0.1, -0.6, 0.4, -0.2, 0.5, 0.0, 0.3, 0.7}};
  W.each_col() /= arma::sum(arma::abs(W), 1);
  arma::mat U = X * W.t();
  U.each_row() -= arma::sum(arma::clamp(W, -arma::datum::inf, 0.0), 1).t();
  arma::vec y(X.n_rows);
  for (arma::uword i = 0; i < X.n_rows; ++i)
    y(i) = branin1(15 * U(i, 0) - 5, 15 * U(i, 1));
  return y;
}

static double hartmann(const arma::rowvec& x, int d) {
  static const double a[4] = {1.0, 1.2, 3.0, 3.2};
  static const double A3[4][3] = {{3, 10, 30}, {0.1, 10, 35}, {3, 10, 30}, {0.1, 10, 35}};
  static const double P3[4][3] = {{0.3689, 0.1170, 0.2673},
                                  {0.4699, 0.4387, 0.7470},
                                  {0.1091, 0.8732, 0.5547},
                                  {0.0381, 0.5743, 0.8828}};
  static const double A6[4][6] = {{10, 3, 17, 3.5, 1.7, 8},
                                  {0.05, 10, 17, 0.1, 8, 14},
                                  {3, 3.5, 1.7, 10, 17, 8},
                                  {17, 8, 0.05, 10, 0.1, 14}};
  static const double P6[4][6] = {{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
                                  {0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
                                  {0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
                                  {0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}};
  double s = 0;
  for (int i = 0; i < 4; ++i) {
    double e = 0;
    for (int j = 0; j < d; ++j) {
      const double A = d == 3 ? A3[i][j] : A6[i][j], P = d == 3 ? P3[i][j] : P6[i][j];
      e += A * (x(j) - P) * (x(j) - P);
    }
    s += a[i] * std::exp(-e);
  }
  return -s;
}

static std::vector<Case> cases() {
  std::vector<Case> c;
  c.push_back({"branin_proj5", 5, 60, branin_proj5});
  c.push_back({"branin2", 2, 30, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& x) { return branin1(15 * x(0) - 5, 15 * x(1)); });
               }});
  c.push_back({"hartmann3", 3, 40,
               [](const arma::mat& X) { return rowmap(X, [](const arma::rowvec& x) { return hartmann(x, 3); }); }});
  c.push_back({"hartmann6", 6, 80,
               [](const arma::mat& X) { return rowmap(X, [](const arma::rowvec& x) { return hartmann(x, 6); }); }});
  c.push_back({"ishigami", 3, 60, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   arma::rowvec x = -M_PI + 2 * M_PI * u;
                   return std::sin(x(0)) + 7 * std::pow(std::sin(x(1)), 2) + 0.1 * std::pow(x(2), 4) * std::sin(x(0));
                 });
               }});
  c.push_back({"borehole", 8, 80, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   const double rw = 0.05 + 0.1 * u(0), r = 100 + 49900 * u(1), Tu = 63070 + 52530 * u(2),
                                Hu = 990 + 120 * u(3), Tl = 63.1 + 52.9 * u(4), Hl = 700 + 120 * u(5),
                                L = 1120 + 560 * u(6), Kw = 9855 + 2190 * u(7);
                   const double lr = std::log(r / rw);
                   return 2 * M_PI * Tu * (Hu - Hl) / (lr * (1 + 2 * L * Tu / (lr * rw * rw * Kw) + Tu / Tl));
                 });
               }});
  c.push_back({"friedman", 5, 60, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& x) {
                   return 10 * std::sin(M_PI * x(0) * x(1)) + 20 * std::pow(x(2) - 0.5, 2) + 10 * x(3) + 5 * x(4);
                 });
               }});
  c.push_back({"rosenbrock4", 4, 60, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   arma::rowvec x = -2 + 4 * u;
                   double s = 0;
                   for (int i = 0; i < 3; ++i)
                     s += 100 * std::pow(x(i + 1) - x(i) * x(i), 2) + std::pow(1 - x(i), 2);
                   return std::log1p(s);
                 });
               }});
  c.push_back({"michalewicz2", 2, 40, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   double s = 0;
                   for (int i = 0; i < 2; ++i) {
                     const double x = M_PI * u(i);
                     s -= std::sin(x) * std::pow(std::sin((i + 1) * x * x / M_PI), 20);
                   }
                   return s;
                 });
               }});
  c.push_back({"goldstein_log", 2, 40, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   const double x = 4 * u(0) - 2, y = 4 * u(1) - 2;
                   const double a = 1 + std::pow(x + y + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y);
                   const double b
                       = 30 + std::pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y);
                   return std::log(a * b);
                 });
               }});
  c.push_back({"otlcircuit", 6, 60, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   const double Rb1 = 50 + 100 * u(0), Rb2 = 25 + 45 * u(1), Rf = 0.5 + 2.5 * u(2), Rc1 = 1.2 + 1.3 * u(3),
                                Rc2 = 0.25 + 0.95 * u(4), b = 50 + 250 * u(5);
                   const double Vb1 = 12 * Rb2 / (Rb1 + Rb2);
                   const double den = b * (Rc2 + 9) + Rf;
                   return (Vb1 + 0.74) * b * (Rc2 + 9) / den + 11.35 * Rf / den + 0.74 * Rf * b * (Rc2 + 9) / (den * Rc1);
                 });
               }});
  c.push_back({"piston", 7, 70, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& u) {
                   const double M = 30 + 30 * u(0), S = 0.005 + 0.015 * u(1), V0 = 0.002 + 0.008 * u(2),
                                k = 1000 + 4000 * u(3), P0 = 90000 + 20000 * u(4), Ta = 290 + 6 * u(5),
                                T0 = 340 + 20 * u(6);
                   const double A = P0 * S + 19.62 * M - k * V0 / S;
                   const double V = S / (2 * k) * (std::sqrt(A * A + 4 * k * P0 * V0 * Ta / T0) - A);
                   return 2 * M_PI * std::sqrt(M / (k + S * S * P0 * V0 * Ta / (T0 * V * V)));
                 });
               }});
  c.push_back({"forrester", 1, 15, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& x) {
                   return std::pow(6 * x(0) - 2, 2) * std::sin(12 * x(0) - 4);
                 });
               }});
  c.push_back({"sin_inter2", 2, 100, [](const arma::mat& X) {
                 return rowmap(X, [](const arma::rowvec& x) { return std::sin(3 * (x(0) + x(1))); });
               }});
  c.push_back({"branin_proj10", 10, 120, branin_proj10});
  return c;
}

int main(int argc, char** argv) {
  const std::string out = argv[1];
  const std::string folds = argv[2];
  const std::string optim = argc > 3 ? argv[3] : "BFGS";
  const std::string filter = argc > 4 ? argv[4] : "";
  const int reps = argc > 5 ? std::atoi(argv[5]) : 3;
  const char* dump = std::getenv("LK_T8_DUMP");
  FILE* fo = std::fopen(out.c_str(), "a");
  for (const auto& cs : cases()) {
    if (!filter.empty() && filter != "all" && cs.name.find(filter) == std::string::npos)
      continue;
    const int rep0 = std::getenv("LK_T8_REP0") ? std::atoi(std::getenv("LK_T8_REP0")) : 0;
    for (int r = rep0; r < reps; ++r) {
      arma::arma_rng::set_seed(1000 + 17 * r + cs.d);
      const arma::mat X = lhs(cs.n, cs.d);
      const arma::vec y = cs.f(X);
      const arma::mat Xt = arma::randu<arma::mat>(1000, cs.d);
      const arma::vec yt = cs.f(Xt);
      const std::vector<arma::uword> hidden = cs.d >= 10 ? std::vector<arma::uword>{16, 8} : std::vector<arma::uword>{8};
      if (dump) {
        FILE* fd = std::fopen(dump, "a");
        std::fprintf(fd, "CASE %s %d %s %s\n", cs.name.c_str(), r, optim.c_str(), folds.c_str());
        std::fclose(fd);
      }
      MLPKriging m(hidden, 2, "tanh", "matern5_2");
      const double c0 = cpu_s();
      const auto w0 = std::chrono::steady_clock::now();
      try {
        m.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", {{"calibration_folds", folds}});
      } catch (const std::exception& e) {
        std::fprintf(stderr, "fit failed %s %d: %s\n", cs.name.c_str(), r, e.what());
        continue;
      }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
      const double cpu = cpu_s() - c0;
      auto [mean, sd, _c, _md, _sd] = m.predict(Xt, true, false, false);
      const double vs = m.var_scale();
      std::fprintf(fo, "FIT %s %d %s %s %.6f %.6f %.17g\n", cs.name.c_str(), r, optim.c_str(), folds.c_str(), wall, cpu,
                   vs);
      // test residuals with the uncalibrated stdev
      for (arma::uword i = 0; i < Xt.n_rows; ++i) {
        const double s0 = sd(i) / std::sqrt(vs);
        std::fprintf(fo, "T %.10g %.10g\n", (yt(i) - mean(i)) / s0, std::log(s0));
      }
      std::fflush(fo);
    }
  }
  std::fclose(fo);
}
