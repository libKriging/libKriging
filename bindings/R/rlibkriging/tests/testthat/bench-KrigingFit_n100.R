library(testthat)

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 100
set.seed(123)
X <- DiceDesign::lhsDesign(n,dimension=3)$design #cbind(runif(n),runif(n))
y <- f(X)
d = ncol(X)

N = 100
times <- list(R=rep(NA, N), cpp=rep(NA, N))

k <- DiceKriging::km(design=X, response=y, covtype = "gauss")
r <- kriging(y, X, "gauss")

for (i in 1:N) {
    times$R[i]   = system.time(
                        k <- DiceKriging::km(design=X,response=y,covtype = "gauss", multistart = 1,control = list(trace=F,maxit=10)) #,lower=rep(0.001,d),upper=rep(2*sqrt(d),d))
                    )
    times$cpp[i] = system.time(
                        r <- kriging(y, X,"gauss","constant",FALSE,"Newton","LL",
                            # to let start optim at same initial point
                            parameters=list(sigma2=0,has_sigma2=FALSE,theta=matrix(k@parinit,ncol=d),has_theta=TRUE))
                    )
}

plot(times$R,times$cpp)
abline(a=0,b=1,col='red')


