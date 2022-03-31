library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
#points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F), nugget=0,nugget.estim = TRUE)
library(rlibkriging)
r <- NuggetKriging(y,X,"gauss","constant",FALSE,"none","LL",
             parameters=list(sigma2=k@covariance@sd2,has_sigma2=TRUE, estim_sigma2=FALSE,
             theta=matrix(k@covariance@range.val),has_theta=TRUE, estim_theta=FALSE,
             nugget=k@covariance@nugget,has_nugget=TRUE, estim_nugget=FALSE))
# m = as.list(r)

ntest <- 100
Xtest <- as.matrix(runif(ntest))
ptest <- DiceKriging::predict(k,Xtest,type="UK",cov.compute = TRUE,checkNames=F)
Yktest <- ptest$mean
sktest <- ptest$sd
cktest <- c(ptest$cov)
Ytest <- predict(r,Xtest,TRUE,TRUE)

plot(f)
points(X,y)
points(Xtest,Yktest,col='blue')
points(Xtest,Ytest$mean,col='red')

precision <- 1e-5
test_that(desc=paste0("pred mean is the same that DiceKriging one:\n ",paste0(collapse=",",Yktest),"\n ",paste0(collapse=",",Ytest$mean)),
          expect_equal(array(Yktest),array(Ytest$mean),tol = precision))

test_that(desc="pred sd is the same that DiceKriging one", 
          expect_equal(array(sktest),array(Ytest$stdev) ,tol = precision))

test_that(desc="pred cov is the same that DiceKriging one", 
          expect_equal(cktest,c(Ytest$cov) ,tol = precision))
