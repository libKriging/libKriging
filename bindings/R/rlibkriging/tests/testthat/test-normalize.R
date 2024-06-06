#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
X10 = 10*X

y = f(X)
y50 = 50*y

context("no normalize")

r_nonorm <- Kriging(y, X, "gauss", normalize=F, optim="BFGS10")
r1050_nonorm <- Kriging(y50, X10, "gauss", normalize=F, optim="BFGS10")

test_that(desc="theta nonorm",
          expect_equal( r_nonorm$theta()*10 , r1050_nonorm$theta() ,tol=0.01))
test_that(desc="beta nonorm",
          expect_equal(r_nonorm$beta()*50 , r1050_nonorm$beta(),tol=0.01))
test_that(desc="sigma2 nonorm",
          expect_equal(r_nonorm$sigma2()*50*50 , r1050_nonorm$sigma2(),tol=0.01))


test_that(desc="predict nonorm",
          expect_equal(lapply(r_nonorm$predict(0.5),function(...){50*...}), r1050_nonorm$predict(10*0.5),tol=0.01))

test_that(desc="simulate nonorm",
          expect_equal(50*r_nonorm$simulate(1,x=0.5), r1050_nonorm$simulate(1,x=10*0.5),tol=0.01))


r_nonorm$update(f(0.5),0.5)
r1050_nonorm$update(50*f(0.5),10*0.5)

test_that(desc="update theta nonorm",
          expect_equal(r_nonorm$theta()*10 , r1050_nonorm$theta(),tol=0.01))
test_that(desc="update beta nonorm",
          expect_equal(r_nonorm$beta()*50 , r1050_nonorm$beta(),tol=0.01))
test_that(desc="update sigma2 nonorm",
          expect_equal(r_nonorm$sigma2()*50*50 , r1050_nonorm$sigma2(),tol=0.01))



context("normalize")

r_norm <- Kriging(y, X, "gauss", normalize=T)
r1050_norm <- Kriging(y50, X10, "gauss", normalize=T)

test_that(desc="theta norm",
          expect_equal(r_norm$theta() , r1050_norm$theta(),tol=0.01))
test_that(desc="beta norm",
          expect_equal(r_norm$beta() , r1050_norm$beta(),tol=0.01))
test_that(desc="sigma2 norm",
          expect_equal(r_norm$sigma2() , r1050_norm$sigma2(),tol=0.01))


test_that(desc="predict norm",
          expect_equal(lapply(r_norm$predict(0.5),function(...){50*...}), r1050_norm$predict(10*0.5),tol=0.01))

test_that(desc="simulate norm",
          expect_equal(50*r_norm$simulate(1,x=0.5), r1050_norm$simulate(1,x=10*0.5),tol=0.01))


plot(seq(0,1,,101),r_norm$simulate(1,seed=123,x=seq(0,1,,101)))
points(X,y,col='red')
plot(seq(0,10,,101),r1050_norm$simulate(1,seed=123,x=seq(0,10,,101)))
points(X10,y50,col='red')

r_norm$update(f(0.5),0.5)
r1050_norm$update(50*f(0.5),10*0.5)

test_that(desc="update theta norm",
          expect_equal(r_norm$theta() , r1050_norm$theta(),tol=0.01))
test_that(desc="update beta norm",
          expect_equal(r_norm$beta() , r1050_norm$beta(),tol=0.01))
test_that(desc="update sigma2 norm",
          expect_equal(r_norm$sigma2() , r1050_norm$sigma2(),tol=0.01))



context("normalize with parameters")

r_norm_param <- Kriging(y, X, "gauss", normalize=T, parameters=list(theta=matrix(0.2),beta=matrix(0.4),sigma2=0.15))
r1050_norm_param <- Kriging(y50, X10, "gauss", normalize=T, parameters=list(theta=matrix(0.2*10),beta=matrix(0.4*50),sigma2=0.15*50*50))

test_that(desc="theta norm_param",
          expect_equal(r_norm_param$theta() , r1050_norm_param$theta(),tol=0.01))
test_that(desc="beta norm_param",
          expect_equal(r_norm_param$beta() , r1050_norm_param$beta(),tol=0.01))
test_that(desc="sigma2 norm_param",
          expect_equal(r_norm_param$sigma2() , r1050_norm_param$sigma2(),tol=0.01))


test_that(desc="predict norm_param",
          expect_equal(lapply(r_norm_param$predict(0.5),function(...){50*...}), r1050_norm_param$predict(10*0.5),tol=0.01))

test_that(desc="simulate norm_param",
          expect_equal(50*r_norm_param$simulate(1,x=0.5), r1050_norm_param$simulate(1,x=10*0.5),tol=0.01))


plot(seq(0,1,,101),r_norm_param$simulate(1,seed=123,x=seq(0,1,,101)))
points(X,y,col='red')
plot(seq(0,10,,101),r1050_norm_param$simulate(1,seed=123,x=seq(0,10,,101)))
points(X10,y50,col='red')

r_norm_param$update(f(0.5),0.5)
r1050_norm_param$update(50*f(0.5),10*0.5)

test_that(desc="update theta norm_param",
          expect_equal(r_norm_param$theta() , r1050_norm_param$theta(),tol=0.01))
test_that(desc="update beta norm_param",
          expect_equal(r_norm_param$beta() , r1050_norm_param$beta(),tol=0.01))
test_that(desc="update sigma2 norm_param",
          expect_equal(r_norm_param$sigma2() , r1050_norm_param$sigma2(),tol=0.01))


