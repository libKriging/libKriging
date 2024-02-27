rlibkriging:::linalg_set_num_nugget(1E-10)
rlibkriging:::linalg_set_chol_warning(TRUE)
default_nugget = rlibkriging:::linalg_get_num_nugget()

#############################################################

context("Chol Nugget")

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 10
d <- 3
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)
r = NULL
try( r <- Kriging(y, X, "gauss","constant",FALSE,"BFGS","LL") )

test_that(desc="Kriging fit with num nugget is passing", expect_true(!is.null(r)))

l=as.list(r)
R = l$T %*% t(l$T) - 1E-10 * diag(nrow(X))
cholR=NULL
try( cholR <- chol(R) ) # Should not fail

test_that(desc="Kriging fit withOUT num nugget would also pass", expect_true(!is.null(cholR)))

rlibkriging:::linalg_set_num_nugget(0.0)
rlibkriging:::linalg_get_num_nugget()

r_nonugget = NULL
try( r_nonugget <- Kriging(y, X, "gauss","constant",FALSE,"BFGS","LL") )

test_that(desc="Kriging fit with num nugget is passing", expect_true(!is.null(r_nonugget)))

l_nonugget=as.list(r_nonugget)
R_nonugget = l_nonugget$T %*% t(l_nonugget$T)
cholR_nonugget=NULL
try( cholR_nonugget <- chol(R_nonugget) ) # Should not fail

test_that(desc="Kriging fit withOUT num nugget would also pass", expect_true(!is.null(cholR_nonugget)))

rlibkriging:::linalg_set_num_nugget(default_nugget)

#############################################################


context("Chol Crash nD")

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
# This will crash "chol(): decomposition failed before adding numerical nugget to R mat
rlibkriging:::linalg_check_chol_rcond(FALSE) # disable failover
try( r <- Kriging(y, X, "gauss","constant",FALSE,"BFGS","LL") )
rlibkriging:::linalg_check_chol_rcond(TRUE) # enable failover

test_that(desc="Kriging fit with num nugget is passing", expect_true(!is.null(r)))

l=as.list(r)
R = l$T %*% t(l$T) - 1E-9 * diag(nrow(X))
cholR=NULL
try( cholR <- chol(R) ) # Must fail

test_that(desc="Kriging fit withOUT num nugget would not pass", expect_true(is.null(cholR)))


#############################################################

context("Nugget has no numerical side effect")

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 10
d <- 3
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)
r = NULL
# This will crash "chol(): decomposition failed before adding numerical nugget to R mat
try( r <- Kriging(y, X, "gauss","constant",FALSE,"BFGS10","LL") )

test_that(desc="Kriging fit with num nugget is passing", expect_true(!is.null(r)))

l=as.list(r)
R = l$T %*% t(l$T) - 1E-10 * diag(nrow(X))
cholR=NULL
try( cholR <- chol(R) ) # Should not fail

test_that(desc="Kriging fit withOUT num nugget would also pass", expect_true(!is.null(cholR)))

rlibkriging:::linalg_set_num_nugget(0.0)
rlibkriging:::linalg_get_num_nugget()

r_nonugget = NULL
# This will crash "chol(): decomposition failed before adding numerical nugget to R mat
try( r_nonugget <- Kriging(y, X, "gauss","constant",FALSE,"BFGS10","LL") )

test_that(desc="Kriging fit with num nugget is passing", expect_true(!is.null(r_nonugget)))

l_nonugget=as.list(r_nonugget)
R_nonugget = l_nonugget$T %*% t(l_nonugget$T)
cholR_nonugget=NULL
try( cholR_nonugget <- chol(R_nonugget) ) # Should not fail

test_that(desc="Kriging fit withOUT num nugget would also pass", expect_true(!is.null(cholR_nonugget)))


test_that(desc="Kriging fit mith and without num nugget are identical", 
expect_true(all(capture.output(print(r_nonugget)) == capture.output(print(r)))))

rlibkriging:::linalg_set_num_nugget(default_nugget)
