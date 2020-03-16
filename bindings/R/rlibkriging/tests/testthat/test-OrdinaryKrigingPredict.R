library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = "gauss")
r <- ordinary_kriging(y,X)
ntest <- 100
Xtest <- as.matrix(runif(ntest))
ptest <- DiceKriging::predict(k,Xtest,type="UK",cov.compute = TRUE)
Yktest <- ptest$mean
sktest <- ptest$sd
cktest <- c(ptest$cov)
Ytest <- ordinary_kriging_predict(r,Xtest,TRUE,TRUE)

precision <- 1e-5
test_that(desc="pred mean is the same that DiceKriging one", 
          expect_true(max(abs(Yktest-Ytest$mean)) < precision))

test_that(desc="pred sd is the same that DiceKriging one", 
          expect_true(max(abs(sktest-Ytest$stdev)) < precision))

test_that(desc="pred cov is the same that DiceKriging one", 
          expect_true(max(abs(cktest-c(Ytest$cov))) < precision))
