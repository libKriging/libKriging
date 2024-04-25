library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
y_o <- f(X_o)
points(X_o, y_o)

lk <- Kriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "constant",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1)))

library(DiceKriging)
dk <- km(response = matrix(y_o, ncol = 1),
              design = matrix(X_o, ncol = 1),
              covtype = "gauss",
              formula = ~1,
              #optim = "none",
              #normalize = TRUE,
              coef.cov = lk$theta()[1,1],
              coef.trend = lk$beta(),
              coef.var = lk$sigma2())

## Predict & simulate

dp = predict(dk, newdata = data.frame(X = seq(0,1,,21)), type="UK", checkNames=FALSE)
lines(seq(0,1,,21),dp$mean,col='blue')
polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(dp$mean+2*dp$sd,rev(dp$mean-2*dp$sd)),col=rgb(0,0,1,0.2),border=NA)

lp = lk$predict(seq(0,1,,21)) # libK predict
lines(seq(0,1,,21),lp$mean,col='red')
polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = lk$simulate(100000, 123, seq(0,1,,21)) # libK simulate
for (i in 1:min(100,ncol(ls))) {
    lines(seq(0,1,,21),ls[,i],col=rgb(1,0,0,.1),lwd=4)
}

ds = simulate(dk, nsim = ncol(ls), newdata = data.frame(X = seq(0,1,,21)), type="UK", checkNames=FALSE, 
cond=TRUE, nugget.sim = 1e-9)
for (i in 1:min(100,nrow(ds))) {
    lines(seq(0,1,,21),ds[i,],col=rgb(0,0,1,.1),lwd=4)
}

for (i in 1:21) {
    if (dp$sd[i] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ds[,i], "pnorm", mean = dp$mean[i],sd = dp$sd[i])$p.value > 0.01))
}

for (i in 1:21) {
    if (lp$stdev[i,] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ls[i,], "pnorm", mean = lp$mean[i,],sd = lp$stdev[i,])$p.value > 0.01))
}

for (i in 1:21) {
    if (dp$sd[i] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ds[,i], ls[i,])$p.value  > 0.01))
}
