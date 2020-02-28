library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = "gauss")
r <- ordinary_kriging(y, X)

precision <- 1e-5
test_that(desc="fit of theta is the same that DiceKriging one", 
         expect_true(abs((ordinary_kriging_model(r)$theta-k@covariance@range.val)/k@covariance@range.val) < precision))

         

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(123)
X <- cbind(runif(n),runif(n))
y = f(X)
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))
r <- ordinary_kriging(y, X)

precision <- 1e-5
test_that(desc="fit of theta 2D is the same that DiceKriging one", 
          expect_true(max(abs((ordinary_kriging_model(r)$theta-k@covariance@range.val)/k@covariance@range.val)) < precision))

