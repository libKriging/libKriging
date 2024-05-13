library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
noise = 0.2^2
set.seed(1234)
y_o <- f(X_o) + rnorm(n, sd = sqrt(noise))
points(X_o, y_o)

lk <- NoiseKriging(y = matrix(y_o, ncol = 1),
                noise = matrix(rep(noise, n), ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "constant",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), sigma2=0.5^2))

library(DiceKriging)
dk <- km(response = matrix(y_o, ncol = 1),
              design = matrix(X_o, ncol = 1),
              covtype = "gauss",
              formula = ~1,
              noise.var = rep(noise, n),
              nugget.estim=FALSE,
              #optim = "none",
              #normalize = TRUE,
              coef.cov = lk$theta()[1,1],
              coef.trend = lk$beta(),
              coef.var = lk$sigma2())

## Predict & simulate
X_n = seq(0,1,,31)

dp = predict(dk, newdata = data.frame(X = X_n), type="UK", checkNames=FALSE)
lines(X_n,dp$mean,col='blue')
polygon(c(X_n,rev(X_n)),c(dp$mean+2*dp$sd,rev(dp$mean-2*dp$sd)),col=rgb(0,0,1,0.2),border=NA)

lp = lk$predict(X_n) # libK predict
lines(X_n,lp$mean,col='red')
polygon(c(X_n,rev(X_n)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = lk$simulate(1000, 123, X_n) # libK simulate
for (i in 1:min(100,ncol(ls))) {
    lines(X_n,ls[,i],col=rgb(1,0,0,.1),lwd=4)
}

ds = simulate(dk, nsim = ncol(ls), newdata = data.frame(X = X_n), type="UK", checkNames=FALSE, 
cond=TRUE, noise.sim = 1e-10)
for (i in 1:min(100,nrow(ds))) {
    lines(X_n,ds[i,],col=rgb(0,0,1,.1),lwd=4)
}

# Failing : DiceKriging is not consistent bw simulate and predict
for (i in 1:length(X_n)) {
    if (dp$sd[i] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("DiceKriging simulate sample ( ~N(",mean(ds[,i]),",",sd(ds[,i]),") ) follows predictive distribution ( =N(",dp$mean[i],",",dp$sd[i],") ) at ",X_n[i]),
        expect_true(ks.test(ds[,i], "pnorm", mean = dp$mean[i],sd = dp$sd[i])$p.value > 0.01))
}

for (i in 1:length(X_n)) {
    if (lp$stdev[i,] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("libKriging simulate sample ( ~N(",mean(ls[i,]),",",sd(ls[i,]),") ) follows predictive distribution ( =N(",lp$mean[i,],",",lp$stdev[i,],") ) at ",X_n[i]),
        expect_true(ks.test(ls[i,], "pnorm", mean = lp$mean[i,],sd = lp$stdev[i,])$p.value > 0.01))
}

# Do not compare with DiceKriging, as it is not consistent bw simulate and predict
for (i in 1:length(X_n)) {
    if (dp$sd[i] > 1e-3) {# otherwise means that density is ~ dirac, so don't test
    plot(density(ds[,i]))
    lines(density(ls[i,]),col='red')
    test_that(desc=paste0("DiceKriging/libKriging simulate samples ( ~N(",mean(ds[,i]),",",sd(ds[,i]),") / ~N(",mean(ls[i,]),",",sd(ls[i,]),") ) matching at ",X_n[i]),
        expect_true(ks.test(ds[,i], ls[i,])$p.value > 0.01))
        print(ks.test(ds[,i], ls[i,])$p.value)
    }
}
