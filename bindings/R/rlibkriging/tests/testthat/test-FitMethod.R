library(testthat)

context("Fit method: explicit exact / Vecchia / Nystrom at every step")

set.seed(6)
X <- matrix(runif(120), ncol = 2)
y <- sin(3 * X[, 1]) + X[, 2]
Xn <- matrix(runif(14), ncol = 2)

test_that("method is stored with its explicit size; objective stays the criterion", {
  expect_equal(Kriging(y, X, "matern5_2")$method(), "exact")
  kv <- Kriging(y, X, "matern5_2", method = "vecchia")
  expect_equal(kv$objective(), "LL")
  expect_equal(kv$method(), "vecchia(30)")
  expect_equal(Kriging(y, X, "matern5_2", method = "nystrom")$method(), "nystrom(50)")
  expect_equal(as.list(kv)$method, "vecchia(30)")
  # LOO / LMP have approximate versions too (without noise model); with
  # objective LL, both approximate methods support the noise models
  expect_equal(Kriging(y, X, "matern5_2", objective = "LOO", method = "vecchia(10)")$objective(), "LOO")
  expect_error(Kriging(y, X, "matern5_2", objective = "LOO", method = "vecchia", noise = "nugget"))
  expect_error(Kriging(y, X, "matern5_2", objective = "LMP", method = "nystrom(10)", noise = "nugget"))
  expect_equal(Kriging(y, X, "matern5_2", method = "nystrom(10)", noise = "nugget")$method(), "nystrom(10)")
  expect_equal(Kriging(y, X, "matern5_2", method = "vecchia", noise = "nugget")$method(), "vecchia(30)")
  expect_error(Kriging(y, X, "matern5_2", method = "Vecchia"))
})

test_that("deprecated LLVecchia / LLNystrom objectives are aliases", {
  expect_warning(ka <- Kriging(y, X, "matern5_2", objective = "LLVecchia(10)"),
                 class = "deprecatedWarning", regexp = 'method="vecchia\\(10\\)"')
  expect_equal(ka$objective(), "LL")
  expect_equal(ka$method(), "vecchia(10)")
  expect_warning(kb <- Kriging(y, X, "matern5_2", objective = "LLNystrom(8)"),
                 class = "deprecatedWarning")
  expect_equal(kb$method(), "nystrom(8)")
  expect_error(suppressWarnings(Kriging(y, X, "matern5_2", objective = "LLVecchia(10)", method = "nystrom")))
  expect_no_warning(Kriging(y, X, "matern5_2", method = "vecchia(10)"))
})

k <- Kriging(kernel = "matern5_2")
k$fit(y, X, method = "vecchia(10)")

test_that("Vecchia fit: explicit method, no implicit switch", {
  expect_equal(k$method(), "vecchia(10)")
  expect_error(k$predict(Xn))                    # exact: no factorization
  expect_error(k$simulate(nsim = 2, seed = 1, x = Xn))
  p <- k$predict(Xn, return_stdev = TRUE, method = "vecchia")
  expect_equal(length(p$mean), 7)
  expect_true(all(is.finite(p$mean)))
  expect_true(all(p$stdev >= 0))
  s <- k$simulate(nsim = 4, seed = 123, x = Xn, method = "vecchia")
  expect_equal(dim(s), c(7, 4))
  expect_equal(dim(k$simulate(nsim = 4, seed = 123, x = Xn, method = "vecchia(5)")), c(7, 4))
  expect_error(k$predict(Xn, return_cov = TRUE, method = "vecchia"))
  expect_error(k$predict(Xn, method = "unknown"))
  expect_true(is.finite(k$logLikelihood(method = "vecchia")))
  expect_equal(k$logLikelihood(method = k$method()), k$logLikelihood(method = "vecchia"))
})

test_that("Vecchia fit: update and save / load", {
  Xu <- matrix(runif(10), ncol = 2)
  yu <- sin(3 * Xu[, 1]) + Xu[, 2]
  k$update(yu, Xu, refit = FALSE)
  expect_equal(k$method(), "vecchia(10)")
  expect_equal(nrow(k$X()), 65)
  expect_true(max(abs(k$predict(Xu, method = "vecchia")$mean - yu)) < 1e-3)
  f <- tempfile(fileext = ".json")
  k$save(f)
  kl <- load(f)
  expect_equal(kl$method(), "vecchia(10)")
  expect_equal(kl$vecchia_neighbors(), 10)
  expect_equal(kl$predict(Xn, method = "vecchia")$mean, k$predict(Xn, method = "vecchia")$mean,
               tolerance = 1e-12)
})

test_that("the exact model at the Vecchia estimate is a separate exact fit", {
  ke <- Kriging(y, X, "matern5_2", optim = "none",
                parameters = list(theta = matrix(k$theta(), nrow = 1), is_theta_estim = FALSE))
  expect_equal(ke$method(), "exact")
  expect_true(is.matrix(ke$predict(Xn, return_cov = TRUE)$cov))
  expect_equal(length(ke$predict(Xn, method = "vecchia")$mean), 7)
  expect_equal(ke$logLikelihood(), ke$logLikelihood(method = "exact"))
})

test_that("Nystrom fits need method nystrom", {
  kn <- Kriging(y, X, "matern5_2", method = "nystrom(8)")
  expect_equal(kn$method(), "nystrom(8)")
  expect_error(kn$predict(Xn))
  expect_equal(length(kn$predict(Xn, method = "nystrom")$mean), 7)
  expect_equal(kn$predict(Xn, method = kn$method())$mean, kn$predict(Xn, method = "nystrom")$mean)
  expect_error(kn$predict(Xn, method = "nystrom(3)"))  # not the fitted rank
  expect_equal(dim(kn$simulate(nsim = 3, seed = 1, x = Xn, method = "nystrom")), c(7, 3))
  expect_true(is.finite(kn$logLikelihood(method = "nystrom")))
  expect_error(kn$logLikelihood(method = "vecchia"))
})

test_that("Nystrom fits with the noise models", {
  set.seed(11)
  yn <- y + rnorm(length(y), sd = 0.05)
  # nugget: estimated noise variance
  knn <- Kriging(yn, X, "matern5_2", noise = "nugget", method = "nystrom(20)")
  expect_equal(knn$method(), "nystrom(20)")
  expect_equal(knn$noise_model(), "nugget")
  expect_true(knn$nugget() > 0)
  expect_error(knn$predict(Xn))
  p <- knn$predict(Xn, method = "nystrom")
  expect_equal(length(p$mean), 7)
  expect_true(all(p$stdev > 0))
  expect_equal(dim(knn$simulate(nsim = 3, seed = 1, x = Xn, method = "nystrom")), c(7, 3))
  expect_true(is.finite(knn$logLikelihood(method = "nystrom")))
  # heterogeneous: known noise vector
  noise <- rep(0.0025, length(yn))
  knh <- Kriging(yn, X, "matern5_2", noise = noise, method = "nystrom(20)")
  expect_equal(knh$noise_model(), "heterogeneous")
  ph <- knh$predict(Xn, method = "nystrom")
  expect_true(all(is.finite(ph$mean)))
  f <- tempfile(fileext = ".json")
  knh$save(f)
  kl <- load(f)
  expect_equal(kl$method(), "nystrom(20)")
  expect_equal(kl$predict(Xn, method = "nystrom")$mean, ph$mean, tolerance = 1e-12)
  # full rank: close to the exact nugget fit
  ke <- Kriging(yn, X, "matern5_2", noise = "nugget")
  kf <- Kriging(yn, X, "matern5_2", noise = "nugget", method = "nystrom(60)")
  expect_true(max(abs(ke$predict(Xn)$mean - kf$predict(Xn, method = "nystrom")$mean)) < 0.1 * sd(yn))
  # Nystrom simulates the latent process only: with_noise raises with it
  expect_equal(dim(knn$simulate(nsim = 3, seed = 1, x = Xn, with_noise = FALSE, method = "nystrom")), c(7, 3))
  expect_error(knn$simulate(nsim = 3, seed = 1, x = Xn, with_noise = TRUE, method = "nystrom"))
  expect_error(knh$simulate(nsim = 3, seed = 1, x = Xn, with_noise = rep(0.1, 7), method = "nystrom"))
})

test_that("LOO / LMP objectives with an approximate method", {
  for (m in c("vecchia(10)", "nystrom(20)")) {
    am <- substr(m, 1, 7)
    kl <- Kriging(y, X, "matern5_2", objective = "LOO", method = m)
    expect_equal(kl$objective(), "LOO")
    expect_equal(kl$method(), m)
    expect_true(is.finite(kl$leaveOneOut(method = am)))
    expect_equal(kl$leaveOneOut(method = m), kl$leaveOneOut(method = am))
    expect_true(is.finite(kl$leaveOneOut()))  # exact LOO at the approximate theta
    loov <- kl$leaveOneOutVec(array(kl$theta()), method = am)
    expect_equal(length(loov$mean), 60)
    expect_equal(mean((loov$mean - y)^2), kl$leaveOneOut(method = am), tolerance = 1e-8)
    kp <- Kriging(y, X, "matern5_2", objective = "LMP", method = m)
    expect_equal(kp$objective(), "LMP")
    expect_true(is.finite(kp$logMargPost(method = am)))
    expect_error(kp$logMargPost(method = "unknown"))
  }
  # with m = n - 1 the Vecchia LOO is the exact one
  ke <- Kriging(y, X, "matern5_2", optim = "none", objective = "LOO", method = "vecchia(59)",
                parameters = list(theta = matrix(c(0.3, 0.3), nrow = 1), is_theta_estim = FALSE))
  expect_equal(ke$leaveOneOut(method = "vecchia"), ke$leaveOneOut(), tolerance = 1e-4)
  expect_equal(ke$logMargPost(method = "vecchia"), ke$logMargPost(), tolerance = 1e-4)
})

test_that("NestedKriging: the common prior from one approximate fit", {
  set.seed(10)
  Xl <- matrix(runif(400), ncol = 2)
  yl <- sin(3 * Xl[, 1]) + Xl[, 2]
  nk <- NestedKriging(yl, Xl, "matern5_2", nb_groups = 3, method = "vecchia(15)")
  expect_equal(nk$objective(), "LL")
  expect_equal(nk$method(), "vecchia(15)")
  expect_true(max(abs(predict(nk, Xl[1:5, , drop = FALSE])$mean - yl[1:5])) < 1e-3)
  expect_equal(NestedKriging(yl, Xl, "matern5_2", nb_groups = 3)$method(), "exact")
})

test_that("method vecchia with noise models (nugget, heterogeneous)", {
  set.seed(11)
  n <- 80
  Xs <- matrix(runif(2 * n), ncol = 2)
  noise <- 0.005 + 0.02 * Xs[, 1]
  ys <- sin(3 * Xs[, 1]) + Xs[, 2] + sqrt(noise) * rnorm(n)
  Xp <- matrix(runif(12), ncol = 2)

  kn <- Kriging(ys, Xs, "matern5_2", noise = "nugget", method = "vecchia(15)")
  expect_equal(kn$method(), "vecchia(15)")
  expect_true(kn$nugget() > 0)
  expect_error(kn$predict(Xp))
  p <- kn$predict(Xp, return_stdev = TRUE, method = "vecchia")
  expect_true(all(is.finite(p$mean)) && all(p$stdev > 0))
  expect_true(is.finite(kn$logLikelihood(method = "vecchia")))
  expect_equal(dim(kn$simulate(nsim = 5, seed = 1, x = Xp, method = "vecchia")), c(6, 5))
  expect_equal(dim(kn$simulate(nsim = 5, seed = 1, x = Xp, with_noise = FALSE, method = "vecchia")), c(6, 5))
  kn$update(ys[1:5] + 0.01, Xs[1:5, , drop = FALSE] + 0.01, refit = FALSE)
  expect_equal(nrow(kn$X()), n + 5)

  kh <- Kriging(ys, Xs, "matern5_2", noise = noise, method = "vecchia(15)")
  expect_equal(kh$method(), "vecchia(15)")
  ph <- kh$predict(Xp, return_stdev = TRUE, method = "vecchia")
  expect_true(all(is.finite(ph$mean)) && all(ph$stdev > 0))
  expect_equal(dim(kh$simulate(nsim = 5, seed = 1, x = Xp, with_noise = rep(0.1, 6), method = "vecchia")),
               c(6, 5))
  kh$update(ys[1:5], noise[1:5], Xs[1:5, , drop = FALSE] + 0.01, FALSE)
  expect_equal(nrow(kh$X()), n + 5)
})
