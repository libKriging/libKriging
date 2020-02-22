f = function(X) apply(X,1,function(x) sum((x-.5)^2))

logn = seq(1,2,by=.1)
times = list(R=rep(NA,length(logn)),cpp=rep(NA,length(logn)))
N = 10000

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- floor(2+i/3)
  
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  X <- upper.tri(X %*% t(X),diag = T)
  y = matrix(runif(n))
  
  times$R[i] = system.time(
    try({for (t in 1:N) k <- backsolve(X, y, upper.tri = T)})
  )[1]
  
  times$cpp[i] = system.time(
    try(r <- bench_solvetri(N,X,y))
  )[1]
}

plot(main = "10000 solvetri",floor(10^logn),log(times$R),ylim=c(log(min(min(times$R),min(times$cpp))),log(max(max(times$R),max(times$cpp)))),xlab="nb points",ylab="log(user_time (s))")
text(20,0,"R")
points(floor(10^logn),log(times$cpp),col='red')
text(80,0,"C++",col = 'red')
