context("LinearAlgebra")

# pack=list.files(file.path("bindings","R"),pattern = ".tar.gz",full.names = T)
# install.packages(pack,repos=NULL)
# library(rlibkriging)
library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

#######################################################################

set.seed(123)
n = 100
X = matrix(rnorm(n*n),n,n)

X1 = t(X) %*% X - .001*diag(n) # will be pos def

L1 = t(chol(X1))

L1_lk = rlibkriging:::linalg_chol_safe(X1)

test_that(desc="Chol R/lK is almost the same", expect_equal(L1, L1_lk, tol=1e-10))

X2 = t(X) %*% X - .01*diag(n) # will not be pos def

L2 = NULL
try(L2 <- t(chol(X2)))

test_that(desc="Chol without nugget does not pass", expect_null(L2))

rlibkriging:::linalg_set_num_nugget(0.001)
rlibkriging:::linalg_set_chol_warning(TRUE)
L2_lk = NULL
L2_lk = rlibkriging:::linalg_chol_safe(X2)
rlibkriging:::linalg_set_num_nugget(1e-15) # default setup

test_that(desc="Chol with nugget passes", expect_false(is.null(L2_lk)))

test_that(desc="Chol with nugget is close to be good",
    expect_equal(L2_lk %*% t(L2_lk) - 6*0.001*diag(n), X2, tol=1e-10))

#######################################################################

rlibkriging:::linalg_set_num_nugget(0) # force NA when chol fails

rcond_chol = function(X) {
    rlibkriging:::linalg_rcond_chol(chol(X))
}
rcond_approx_chol = function(X) {
    rlibkriging:::linalg_rcond_approx_chol(chol(X))
}
rcond = function(X) {
    1/kappa(X, norm='2')
}

set.seed(123)
r = array(NA, 100)
r_chol = array(NA,length(r))
ra_chol = array(NA,length(r))
for (i in 1:length(r)) {
    n = 100
    X = matrix(rnorm(n*n),n,n)
    X = t(X) %*% X + n*rnorm(1,0,1)*diag(n)
    try(r[i] <- rcond((X)))
    try(r_chol[i] <- rcond_chol((X)))
    try(ra_chol[i] <- rcond_approx_chol((X)))
}

plot(log10(r),log10(r_chol),col='red',ylim=range(c(range(log10(r_chol),na.rm=T),range(log10(ra_chol),na.rm=T))))
points(log10(r),log10(ra_chol),col='orange')
abline(a=0,b=1)

rlibkriging:::linalg_set_num_nugget(1e-15) # default setup

#######################################################################

n = 100
R = matrix(rnorm(n*n),n,n)
R = t(R) %*% R

## we just want a covar (sym pos def) matrix
no = n-30
Roo = R[1:no,1:no]
Too = t(chol(Roo))

test_that(desc="Full chol equals incremented chol by block",
          expect_equal(t(chol(R)), rlibkriging:::linalg_chol_block(R,Too,Roo), tol=1e-9))
