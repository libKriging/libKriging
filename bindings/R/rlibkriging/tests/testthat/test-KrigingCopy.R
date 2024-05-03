context("test_copied_kriging_returns_same_result")
    X = matrix(c(0.0, 0.2, 0.5, 0.8, 1.0))
    f = function(x) (1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x ^ 5 + 0.7))
    y = f(X)

    rl1 = Kriging(y, X, "gauss", parameters=list(sigma2=1, is_theta_estim=TRUE))
    print(rl1)

    rl2 = rl1$copy()  # true copy not reference copy
    print(rl2)

    test_that(desc="not same object reference",
          expect_false(identical(rl1,rl2)))  # not same object reference                  

    x = seq(0, 1, 1 / 99)

    p1 = rl1$predict(x, TRUE, TRUE, TRUE)
    p1 = list(mean=p1[1], stdev=p1[2], cov=p1[3], mean_deriv=p1[4], stdev_deriv=na.omit(p1[5]))

    p2 = rl2$predict(x, TRUE, TRUE, TRUE)
    p2 =  list(mean=p2[1], stdev=p2[2], cov=p2[3], mean_deriv=p2[4], stdev_deriv=na.omit(p2[5]))

    test_that(desc="mean",expect_equal(p1["mean"], p2["mean"]))
    test_that(desc="stdev",expect_equal(p1["stdev"], p2["stdev"]))
    test_that(desc="cov",expect_equal(p1["cov"], p2["cov"]))
    test_that(desc="mean_deriv",expect_equal(p1["mean_deriv"], p2["mean_deriv"]))
    test_that(desc="stdev_deriv",expect_equal(p1["stdev_deriv"], p2["stdev_deriv"]))


context("test_copied_and_changed_kriging_returns_different_result")
    X = matrix(c(0.0, 0.2, 0.5, 0.8, 1.0))
    f = function(x) (1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x ^ 5 + 0.7))
    y = f(X)

    rl1 = Kriging(y, X, "gauss", parameters=list(sigma2=1, is_theta_estim=TRUE))
    print(rl1)

    rl2 = rl1$copy()  # true copy not reference copy
    print(rl2)

    test_that(desc="not same object reference",
          expect_false(identical(rl1,rl2)))  # not same object reference                  

    x = seq(0, 1, 1 / 99)

    p1 = rl1$predict(x, TRUE, FALSE, FALSE)
    p1 = list(mean=p1[1], stdev=p1[2], cov=p1[3], mean_deriv=p1[4], stdev_deriv=p1[5])

    rl2$update(f(0.6), 0.6, TRUE)
    p2 = rl2$predict(x, TRUE, FALSE, FALSE)
    p2 =  list(mean=p2[1], stdev=p2[2], cov=p2[3], mean_deriv=p2[4], stdev_deriv=p2[5])

    test_that(desc="mean",expect_false(identical(p1["mean"], p2["mean"])))
    test_that(desc="stdev",expect_false(identical(p1["stdev"], p2["stdev"])))
