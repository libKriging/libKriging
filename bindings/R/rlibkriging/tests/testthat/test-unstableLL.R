# pack=list.files(file.path("bindings","R"),pattern = ".tar.gz",full.names = T)
# install.packages(pack,repos=NULL)
# library(rlibkriging)
#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(testthat)
default_rcond_checked = rlibkriging:::linalg_chol_rcond_checked()
default_num_nugget = rlibkriging:::linalg_get_num_nugget()

context("Fit: unstable LL (long range with 1D Gauss kernel)")

# 1D function, small design, but not stationary
X <- as.matrix(c(0.0, 0.25, 0.33, 0.45, 0.5, 0.75, 1.0))
f <- function(x) 1 - 1 / 2 * (sin(4 * x) / (1 + x) + 2 * cos(12 * x) * x^6 + 0.7)
y <- f(X)  #+ 0.5*rnorm(nrow(X))

#plot(f)
#points(X, y)

rlibkriging:::linalg_check_chol_rcond(FALSE)
rlibkriging:::linalg_set_num_nugget(1e-15)
rlibkriging:::linalg_set_chol_warning(TRUE)
#rlibkriging:::optim_log(4)

# bad init theta value
# Build Kriging (https://libkriging.readthedocs.io/en/latest/math/KrigingModels.html)
k <- Kriging(y, X, kernel="gauss", optim="none", parameters= list(theta=matrix(9.0), sigma2=1e10))
plot(f)
points(X,y)
#DiceView::sectionview(k,add=TRUE)
xx=sort(c(seq(0,1,,101),X))
lines(xx, k$predict(xx)$mean)

# Build Kriging (https://libkriging.readthedocs.io/en/latest/math/KrigingModels.html)
#rlibkriging:::optim_log(4)
rlibkriging:::linalg_check_chol_rcond(TRUE)
k <- Kriging(y, X, kernel="gauss", optim="BFGS")
#DiceView::sectionview(k,add=TRUE,col_surf='orange')

# plots
.t=seq(0,10,,101)
rlibkriging:::linalg_check_chol_rcond(TRUE)
plot(.t, k$logLikelihoodFun(.t)$logLikelihood)
for (t in .t) {
    arrows(
        t,     k$logLikelihoodFun(t)$logLikelihood,
        t-0.2, k$logLikelihoodFun(t)$logLikelihood-0.2*k$logLikelihoodFun(t,return_grad=TRUE)$logLikelihoodGrad, 
        col='black')
}
rlibkriging:::linalg_check_chol_rcond(FALSE)
lines(.t, k$logLikelihoodFun(.t)$logLikelihood,col='blue')
for (t in .t) {
    arrows(
        t,     k$logLikelihoodFun(t)$logLikelihood,
        t-0.21, k$logLikelihoodFun(t)$logLikelihood-0.21*k$logLikelihoodFun(t,return_grad=TRUE)$logLikelihoodGrad, 
        col='blue')
}
#rlibkriging:::covariance_use_approx_singular(FALSE)
#rlibkriging:::linalg_set_num_nugget(0)

abline(v=k$theta())
abline(h=k$logLikelihood())

rcond_approx_R = function(theta) {
    k_tmp = NULL
    try(k_tmp <- Kriging(y, X, kernel="gauss", optim="none",parameters=list(theta=matrix(theta))))
    if (is.null(k_tmp)) return(NA)
    T=k_tmp$T()
    R = (T) %*% t(T)
    rlibkriging:::linalg_rcond_approx_chol(T)
}
lines(.t, log(Vectorize(rcond_approx_R)(.t)),col='orange')
abline(h=log(1e-10),col='orange')
rcond_R = function(theta) {
    k_tmp = NULL
    try(k_tmp <- Kriging(y, X, kernel="gauss", optim="none",parameters=list(theta=matrix(theta))))
    if (is.null(k_tmp)) return(NA)
    T=k_tmp$T()
    R = (T) %*% t(T)
    rlibkriging:::linalg_rcond_chol(T)
}
lines(.t, log(Vectorize(rcond_R)(.t)),col='red')
abline(h=log(1e-18),col='red')
# rcond = function(theta) {
#     k_tmp = NULL
#     try(k_tmp <- Kriging(y, X, kernel="gauss", optim="none",parameters=list(theta=matrix(theta))))
#     if (is.null(k_tmp)) return(NA)
#     T=k_tmp$T()
#     R = (T) %*% t(T)
#     1/kappa(R,norm='2')
# }
# lines(.t, log(Vectorize(rcond)(.t)),col='pink')

k10 <- Kriging(y, X, kernel="gauss", optim="BFGS10")

abline(v=k10$theta(), col='red')
abline(h=k10$logLikelihood(), col='red')

test_that(desc=paste0("LL / Fit: unstable LL fixed using rcond failover (linalg_get_num_nugget:", rlibkriging:::linalg_get_num_nugget(), ")"),
          expect_equal(k$theta(), k10$theta(), tol=1e-4))


rlibkriging:::linalg_check_chol_rcond(default_rcond_checked)
rlibkriging:::linalg_set_num_nugget(default_num_nugget)