rlibkriging:::linalg_check_chol_rcond(FALSE)
rlibkriging:::linalg_set_chol_warning(TRUE)
rlibkriging:::linalg_set_num_nugget(1e-10)

########################################################

context("Chol incremental by block")

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 100
d <- 3
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)
r = NULL
try( r <- Kriging(y, X, "gauss","constant",FALSE,"BFGS","LL") )

## we just want a covar (sym pos def) matrix
R = (r$T()) %*% t(r$T())
no = n-30
Roo = R[1:no,1:no]
Too = t(chol(Roo))

test_that(desc="Full chol equals incremented chol by block",
          expect_equal(r$T(), rlibkriging:::linalg_chol_block(R,Too), tol=1e-7))

########################################################

context("increment Kriging")

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 1000
d <- 3
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)
r = NULL
try( r <- Kriging(y, X, "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0.01))) )

no = floor(n*0.7)
try( ro <- Kriging(y[1:no], X[1:no,], "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0.01))) )
# update with new points, compte LL but no fit (since optim=none)
rlibkriging:::kriging_assimilate(ro,y[(no+1):n], X[(no+1):n,])

test_that(desc="Updated (no refit) Kriging equals Kriging with all data",
          expect_equal(ro$T(), r$T(), tol=1e-5))

# m1 = microbenchmark::microbenchmark(
#   r <- Kriging(y, X, "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0.01))),
#   times=100
# )
# 
# m2 = microbenchmark::microbenchmark(
#   {
#     ro <- Kriging(y[1:no], X[1:no,], "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0.01)))
#     rlibkriging:::kriging_assimilate(ro,y[(no+1):n], X[(no+1):n,])
#   },
#   times=100
# )
