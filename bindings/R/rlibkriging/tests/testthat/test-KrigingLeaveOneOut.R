library(testthat)

for (kernel in c("gauss","exp","matern3_2","matern5_2")) {
# kernel = "gauss"
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F))
ll = function(theta) DiceKriging::leaveOneOutFun(theta,k)

plot(Vectorize(ll),ylab="LL",xlab="theta",xlim=c(0.01,1))
for (x in seq(0.01,1,,11)){
  envx = new.env()
  llx = DiceKriging::leaveOneOutFun(x,k,envx)
  gllx = DiceKriging::leaveOneOutGrad(x,k,envx)
  arrows(x,llx,x+.1,llx+.1*gllx)
}

r <- Kriging(y, X, kernel)
ll2 = function(theta) leaveOneOut(r,theta)$leaveOneOut[1]
# plot(Vectorize(ll2),col='red',add=T) # FIXME fails with "error: chol(): decomposition failed"
for (x in seq(0.01,1,,11)){
  envx = new.env()
  ll2x = leaveOneOut(r,x)$leaveOneOut[1]
  gll2x = leaveOneOut(r,x,grad=T)$leaveOneOutGrad
  arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
}

precision <- 1e-8  # the following tests should work with it, since the computations are analytical
x=.5
xenv=new.env()
test_that(desc="leaveOneOut is the same that DiceKriging one",
         expect_equal(leaveOneOut(r,x)$leaveOneOut[1],DiceKriging::leaveOneOutFun(x,k,xenv),tolerance = precision))

test_that(desc="leaveOneOut Grad is the same that DiceKriging one",
          expect_equal(leaveOneOut(r,x,grad=T)$leaveOneOutGrad,DiceKriging::leaveOneOutGrad(x,k,xenv),tolerance= precision))
}


