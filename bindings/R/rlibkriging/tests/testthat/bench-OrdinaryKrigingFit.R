library(foreach)
registerDoSEQ()

f = function(X) apply(X,1,function(x) sum(x^2))

logn = seq(1,1.5,by=.1)
times = list(R=rep(NA,length(logn)),cpp=rep(NA,length(logn)))

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- floor(1+i)
    
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y = f(X)
  
  times$R[i] = system.time(
    try(k <- DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F), multistart = 10))
  )
  
  times$cpp[i] = system.time(
    try(r <- ordinary_kriging(y, X))
  )
  
  print(max(abs(ordinary_kriging_model(r)$theta-k@covariance@range.val)))
}

plot(floor(10^logn),times$R,ylim=c(0,max(max(times$R),max(times$cpp))))
points(floor(10^logn),times$cpp,col='red')
