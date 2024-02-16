library(rlibkriging, lib.loc="bindings/R/Rlibs")

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
ro = NULL
no = n-30

tr = rep(NA,100)
tro = rep(NA,length(tr))

for (i in 1:length(tr)) {
  t0 = Sys.time() 
   r <- Kriging(y, X, "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0))) 
  tr[i] = Sys.time() - t0

   ro <- Kriging(y[1:no], X[1:no,], "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), sigma2 = 0.1, beta = matrix(0))) 
  # update with new points, compte LL but no fit (since optim=none)
  t0 = Sys.time()
  ro$update_nofit(y[(no+1):n], X[(no+1):n,])
  tro[i] = Sys.time() - t0
  
  if (max(ro$T() - r$T())>1e-9) stop("Not matching Cholesky !")
}

plot(tr,tro)
abline(a=0,b=1)
