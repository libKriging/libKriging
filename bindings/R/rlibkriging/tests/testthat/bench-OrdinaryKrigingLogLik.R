library(testthat)

f = function(X) apply(X,1,function(x) sum(x^2))
n <- 2000
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n))
y = f(X)

xx = seq(0.01,1,,11)
times = list(R_ll=rep(NA,length(xx)),R_gll=rep(NA,length(xx)),cpp_ll=rep(NA,length(xx)),cpp_gll=rep(NA,length(xx)))


k = DiceKriging::km(design=X,response=y,covtype = "gauss")
i=1
for (x in xx){
  envx = new.env()
  times$R_ll[i]=system.time(llx <- DiceKriging::logLikFun(rep(x,3),k,envx))
  times$R_gll[i]=system.time(gllx <- DiceKriging::logLikGrad(rep(x,3),k,envx))
  i=i+1
}

r <- ordinary_kriging(y, X)
i=1
for (x in xx){
  times$cpp_ll[i]=system.time(ll2x <- ordinary_kriging_loglikelihood(r,rep(x,3)))
  times$cpp_gll[i]=system.time(gll2x <- ordinary_kriging_loglikelihoodgrad(r,rep(x,3)))
  i=i+1
}
