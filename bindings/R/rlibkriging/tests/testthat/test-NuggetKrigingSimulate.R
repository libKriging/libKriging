library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
nugget = 0.1^2
set.seed(1234)
y_o <- f(X_o) #+ rnorm(n, sd = sqrt(nugget))
points(X_o, y_o)

lk <- NuggetKriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "constant",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), nugget=nugget, sigma2=0.5^2))

library(DiceKriging)
dk <- km(response = matrix(y_o, ncol = 1),
              design = matrix(X_o, ncol = 1),
              covtype = "gauss",
              formula = ~1,
              nugget = nugget,
              nugget.estim=FALSE,
              #optim = "none",
              #normalize = TRUE,
              coef.cov = lk$theta()[1,1],
              coef.trend = lk$beta(),
              coef.var = lk$sigma2())

test_that("DiceKriging/libKriging T matrix is the same", {
    expect_equal(t(dk@T) , lk$T()*sqrt(lk$sigma2()+lk$nugget()))
})

test_that("DiceKriging/libKriging covariance matrix is the same", {
    expect_equal((lk$covMat(matrix(X_o,ncol=1),matrix(X_o,ncol=1))*lk$sigma2() + diag(nugget,n)), covMatrix(dk@covariance,dk@X)$C
    )
})

## Predict & simulate
X_n = seq(0.06,0.97,,11) #unique(sort(c(X_o,seq(0,1,,21))))

## Check that DiceKriging and libKriging matches (at factor alpha)

# DiceKriging / kmStruct.R / simulate.km :
object = dk
newdata = matrix(X_n, ncol = 1)
Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)          ## size n x m
Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE)

# libKriging / NuggetKriging.cpp / simulate(...,with_nugget=TRUE) :
alpha = lk$sigma2()/(lk$sigma2()+lk$nugget())
R_on = lk$covMat(object@X, newdata) * alpha

test_that("DiceKriging/libKriging R_on matrix is the same", {
    expect_equal(Sigma21, R_on * (lk$sigma2()+lk$nugget()))
})

Rstar_on = backsolve(lk$T(), R_on, upper.tri = FALSE)

test_that("DiceKriging/libKriging Rstar_on matrix is the same", {
    expect_equal(Rstar_on * sqrt(lk$sigma2()+lk$nugget()), Tinv.Sigma21)
})

R_nn = lk$covMat(newdata, newdata) * alpha
diag(R_nn) = 1


# libK
Sigma_nKo = R_nn - crossprod(Rstar_on, Rstar_on)
chol(Sigma_nKo) # -> OK
# DiceKriging
Sigma_cond = covMatrix(object@covariance, newdata)[[1]] - t(Tinv.Sigma21) %*% Tinv.Sigma21
chol(Sigma_cond) # -> OK

test_that("DiceKriging/libKriging Sigma_cond matrix is the same", {
    expect_equal(Sigma_nKo *(lk$sigma2()+lk$nugget()), Sigma_cond)
})


# libKriging / NuggetKriging.cpp / simulate(...,with_nugget=FALSE) :
R_on = lk$covMat(object@X, newdata) #* alpha
Rstar_on = backsolve(lk$T(), R_on, upper.tri = FALSE)
R_nn = lk$covMat(newdata, newdata) #* alpha
diag(R_nn) = 1
Sigma_nKo = R_nn - crossprod(Rstar_on, Rstar_on)
chol(Sigma_nKo) # -> ERROR





















dp = predict(dk, newdata = data.frame(X = X_n), type="UK", checkNames=FALSE)
lines(X_n,dp$mean,col='blue')
polygon(c(X_n,rev(X_n)),c(dp$mean+2*dp$sd,rev(dp$mean-2*dp$sd)),col=rgb(0,0,1,0.2),border=NA)

lp = lk$predict(X_n) # libK predict
lines(X_n,lp$mean,col='red')
polygon(c(X_n,rev(X_n)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = lk$simulate(1000, 123, X_n, with_nugget=FALSE) # libK simulate
for (i in 1:min(100,ncol(ls))) {
    lines(X_n,ls[,i],col=rgb(1,0,0,.1),lwd=4)
}

ds = simulate(dk, nsim = ncol(ls), newdata = data.frame(X = X_n), type="UK", checkNames=FALSE, 
cond=TRUE, nugget.sim = 1e-10)
for (i in 1:min(100,nrow(ds))) {
    lines(X_n,ds[i,],col=rgb(0,0,1,.1),lwd=4)
}

for (i in 1:length(X_n)) {
    if (dp$sd[i] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("DiceKriging simulate sample ( ~N(",mean(ds[,i]),",",sd(ds[,i]),") ) follows predictive distribution ( =N(",dp$mean[i],",",dp$sd[i],") ) at ",X_n[i]),
        expect_true(ks.test(ds[,i], "pnorm", mean = dp$mean[i],sd = dp$sd[i])$p.value > 0.001))
}

for (i in 1:length(X_n)) {
    if (lp$stdev[i,] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("libKriging simulate sample ( ~N(",mean(ls[i,]),",",sd(ls[i,]),") ) follows predictive distribution ( =N(",lp$mean[i,],",",lp$stdev[i,],") ) at ",X_n[i]),
        expect_true(ks.test(ls[i,], "pnorm", mean = lp$mean[i,],sd = lp$stdev[i,])$p.value > 0.01))
}

for (i in 1:length(X_n)) {
    if (dp$sd[i] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("DiceKriging/libKriging simulate samples ( ~N(",mean(ds[,i]),",",sd(ds[,i]),") / ~N(",mean(ls[i,]),",",sd(ls[i,]),") ) matching at ",X_n[i]),
        expect_true(ks.test(ds[,i], ls[i,])$p.value > 0.001))
        print(ks.test(ds[,i], ls[i,])$p.value)
}
