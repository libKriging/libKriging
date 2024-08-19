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
n_u = 50
X_u = matrix(runif(n_u*d),ncol=d)
y_u = f(X_u)

nsim = 1000
n_n = 101
X_n = matrix(runif(n_n*d),ncol=d)

r <- Kriging(y, X, "matern3_2", # "gauss" is too unstable
    "constant",FALSE,"none","LL", parameters=list(theta = matrix(.2,ncol=d), sigma2 = 1, beta = matrix(0)))

rs = NULL
rs = r$simulate(nsim, 123, X_n, will_update=TRUE)

if (d==1) {
    .x=matrix(seq(0,1,,101))
    plot(.x,f(.x),type='l')
    points(X,y)
    #points(X_u,y_u,col='red')
    iX_n = sort(X_n,index.return=TRUE)$ix
    for (i in 1:nsim) {
        lines(X_n[iX_n],rs[iX_n,i],col=rgb(1,0,0,.1))
    }
    rc = copy(r)
    rc$update(y_u, X_u, refit=FALSE)
    rus = NULL
    rus = rc$simulate(nsim, 123, X_n)
    rsu = NULL
    rsu = r$update_simulate(y_u, X_u)

    points(X_u,y_u,col='red',pch=20)
    for (j in 1:nsim) {
        lines(X_n[iX_n],rus[iX_n,j],type='l',col=rgb(0,0,1,.1))
        lines(X_n[iX_n],rsu[iX_n,j],type='l',col=rgb(0,1,0,.1))
    }
}

tus = rep(NA,100)
tsu = rep(NA,length(tus))

for (i in 1:length(tus)) {
    # slightly change output
    y_u_i = y_u #+ 0.1 * rnorm(n_u)

    rc = copy(r)

  t0 = Sys.time()
    rc$update(y_u_i, X_u, refit=FALSE)
    rus = rc$simulate(nsim, 123, X_n)
  tus[i] = Sys.time() - t0

  t0 = Sys.time()
    rsu = r$update_simulate(y_u_i, X_u)
  tsu[i] = Sys.time() - t0
  
  for (j in 1:n_n) {
    if (ks.test(rus[j,],rsu[j,])$p.value < 0.05) {
        plot(density(rus[j,]))
        lines(density(rsu[j,]),col='red')
        stop("Not matching simulate sample !")
    }
  }
}

plot(tus,log(tsu/tus),xlab="update+simulate",ylab="log( update_simulate / update+simulate )",type='p')
