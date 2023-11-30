context("Fit: unstable LL (long range with 1D Gauss kernel)")

# 1D function, small design, but not stationary
X <- as.matrix(c(0.0, 0.25, 0.33, 0.45, 0.5, 0.75, 1.0))
f <- function(x) 1 - 1 / 2 * (sin(4 * x) / (1 + x) + 2 * cos(12 * x) * x^6 + 0.7)
y <- f(X)  #+ 0.5*rnorm(nrow(X))

#plot(f)
#points(X, y)

library(rlibkriging)
#rlibkriging:::optim_log(4)
# Build Kriging (https://libkriging.readthedocs.io/en/latest/math/KrigingModels.html)
k <- Kriging(y, X, kernel="gauss", optim="BFGS")
# kernel: https://libkriging.readthedocs.io/en/latest/math/kernel.html
# regmodel: https://libkriging.readthedocs.io/en/latest/math/trend.html
# parameters: https://libkriging.readthedocs.io/en/latest/math/parameters.html
# print(k)
k10 <- Kriging(y, X, kernel="gauss", optim="BFGS10")

test_that(desc="LL / Fit: unstable LL fixed using rcond failover",
          expect_equal(k$theta(), k10$theta(), tol=1e-5))
