library(testthat)
#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#rlibkriging:::optim_log(2)
#rlibkriging:::optim_use_reparametrize(FALSE)
#rlibkriging:::optim_set_theta_lower_factor(0.02)


f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
X10 = 10*X

y = f(X)
y10 = 10*y

library(rlibkriging)


context("no normalize")

r_nonorm <- Kriging(y, X, "gauss", normalize=F)
r10_nonorm <- Kriging(y10, X10, "gauss", normalize=F)

test_that(desc="theta nonorm",
          expect_equal( r_nonorm$theta()*10 , r10_nonorm$theta() ,tol=0.01))
test_that(desc="beta nonorm",
          expect_equal(r_nonorm$beta()*10 , r10_nonorm$beta(),tol=0.01))
test_that(desc="sigma2 nonorm",
          expect_equal(r_nonorm$sigma2()*100 , r10_nonorm$sigma2(),tol=0.01))


test_that(desc="predict nonorm10 = 10*nonorm",
          expect_equal(lapply(r_nonorm$predict(0.5),function(...)10*...), r10_nonorm$predict(10*0.5),tol=0.01))

test_that(desc="simulate nonorm10 = 10*nonorm",
          expect_equal(10*r_nonorm$simulate(1,x=0.5), r10_nonorm$simulate(1,x=10*0.5),tol=0.01))


r_nonorm$update(newX=0.5,newy=f(0.5))
r10_nonorm$update(newX=10*0.5,newy=10*f(0.5))


test_that(desc="update theta nonorm10 = 10*nonorm",
          expect_equal(r_nonorm$theta()*10 , r10_nonorm$theta(),tol=0.01))
test_that(desc="update beta nonorm10 = 10*nonorm",
          expect_equal(r_nonorm$beta()*10 , r10_nonorm$beta(),tol=0.01))
test_that(desc="update sigma2 nonorm10 = 10*nonorm",
          expect_equal(r_nonorm$sigma2()*100 , r10_nonorm$sigma2(),tol=0.01))



context("normalize")

r_norm <- Kriging(y, X, "gauss", normalize=T)
r10_norm <- Kriging(y10, X10, "gauss", normalize=T)

test_that(desc="theta norm",
          expect_equal(r_norm$theta()*10 , r10_norm$theta(),tol=0.01))
test_that(desc="beta norm",
          expect_equal(r_norm$beta()*10 , r10_norm$beta(),tol=0.01))
test_that(desc="sigma2 norm",
          expect_equal(r_norm$sigma2()*100 , r10_norm$sigma2(),tol=0.01))


test_that(desc="predict norm10 = 10*norm",
          expect_equal(lapply(r_norm$predict(0.5),function(...)10*...), r10_norm$predict(10*0.5),tol=0.01))

test_that(desc="simulate norm10 = 10*norm",
          expect_equal(10*r_norm$simulate(1,x=0.5), r10_norm$simulate(1,x=10*0.5),tol=0.01))


plot(seq(0,1,,101),r_norm$simulate(1,seed=123,x=seq(0,1,,101)))
points(X,y,col='red')
plot(seq(0,10,,101),r10_norm$simulate(1,seed=123,x=seq(0,10,,101)))
points(X10,y10,col='red')

r_norm$update(newX=0.5,newy=f(0.5))
r10_norm$update(newX=10*0.5,newy=10*f(0.5))

test_that(desc="update theta norm10 = 10*norm",
          expect_equal(r_norm$theta()*10 , r10_norm$theta(),tol=0.01))
test_that(desc="update beta norm10 = 10*norm",
          expect_equal(r_norm$beta()*10 , r10_norm$beta(),tol=0.01))
test_that(desc="update sigma2 norm10 = 10*norm",
          expect_equal(r_norm$sigma2()*100 , r10_norm$sigma2(),tol=0.01))


