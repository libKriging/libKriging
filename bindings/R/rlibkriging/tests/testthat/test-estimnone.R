library(testthat)
#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#rlibkriging:::optim_log(2)
#rlibkriging:::optim_use_reparametrize(FALSE)
#rlibkriging:::optim_set_theta_lower_factor(0.02)


f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)

library(rlibkriging)


context("Kriging")

r_noestim <- Kriging(y, X, "gauss", optim="none", parameters=list(theta= matrix(0.1) ,sigma2= 0.01 ,beta= matrix(0.123)))
print(r_noestim)
test_that(desc="theta noestim",
          expect_equal( r_noestim$theta()[1] , 0.1 ,tol=1E-10))
test_that(desc="sigma2 noestim",
          expect_equal( r_noestim$sigma2() , 0.01 ,tol=1E-10))
test_that(desc="beta noestim",
          expect_equal( r_noestim$beta()[1] , 0.123 ,tol=1E-10))


context("NuggetKriging")

rnu_noestim <- NuggetKriging(y, X, "gauss", optim="none", parameters=list(theta= matrix(0.1) ,sigma2= 0.01 ,beta= matrix(0.123), nugget= 0.0456))
print(rnu_noestim)
test_that(desc="theta noestim",
          expect_equal( rnu_noestim$theta()[1] , 0.1 ,tol=1E-10))
test_that(desc="sigma2 noestim",
          expect_equal( rnu_noestim$sigma2() , 0.01 ,tol=1E-10))
test_that(desc="beta noestim",
          expect_equal( rnu_noestim$beta()[1] , 0.123 ,tol=1E-10))
test_that(desc="nugget noestim",
          expect_equal( rnu_noestim$nugget() , 0.0456 ,tol=1E-10))

context("NoiseKriging")

rno_noestim <- NoiseKriging(y, rep(0.05,nrow(X)) , X, "gauss", optim="none", parameters=list(theta= matrix(0.1) ,sigma2= 0.01 ,beta= matrix(0.123)))
print(rno_noestim)
test_that(desc="theta noestim",
          expect_equal( rno_noestim$theta()[1] , 0.1 ,tol=1E-10))
test_that(desc="sigma2 noestim",
          expect_equal( rno_noestim$sigma2() , 0.01 ,tol=1E-10))
test_that(desc="beta noestim",
          expect_equal( rno_noestim$beta()[1] , 0.123 ,tol=1E-10))
