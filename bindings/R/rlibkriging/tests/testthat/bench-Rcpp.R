library(rlibkriging)

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))

logn <- seq(1.1, 2.5, by=.1)
times <- list(chol_cpp_user=rep(NA, length(logn)),
              chol_cpp_elapsed=rep(NA, length(logn)),
              chol_icpp_elapsed=rep(NA, length(logn)),
              nochol_cpp_user=rep(NA, length(logn)),
              nochol_cpp_elapsed=rep(NA, length(logn)),
              nochol_icpp_elapsed=rep(NA, length(logn)))
times.n = 1

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1+floor(log(n)) #floor(2+i/3)

  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y <- f(X)

  T = system.time(
    try(for (j in 1:times.n) r <- bench_rcpp_link(y, X, 1, 2000))
  ) # 1: user time, 2: system time, 3:elapsed time
  times$chol_cpp_user[i] <- T[1]
  times$chol_cpp_elapsed[i] <- T[3]
  times$chol_icpp_elapsed[i] <- attr(r,'time')

  T = system.time(
    try(for (j in 1:times.n) r <- bench_rcpp_link(y, X, 2000, 0))
  ) # 1: user time, 2: system time, 3:elapsed time
  times$nochol_cpp_user[i] <- T[1]
  times$nochol_cpp_elapsed[i] <- T[3]
  times$nochol_icpp_elapsed[i] <- attr(r,'time')
}

#plot(floor(10^logn),log(times$cpp),ylim=c(log(min(min(times$cpp,na.rm = T),min(times$icpp,na.rm = T))),
#                                          log(max(max(times$cpp,na.rm = T),max(times$icpp,na.rm = T)))),
#     xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
#points(floor(10^logn),log(times$icpp),col='green')

plot(floor(10^logn),times$chol_cpp_elapsed, col='grey', pch=3,
     ylim=c(min(min(times$chol_cpp_elapsed,na.rm = T),min(times$chol_cpp_user,na.rm = T)),
            max(max(times$chol_cpp_elapsed,na.rm = T),max(times$chol_cpp_user,na.rm = T))),
     xlab="nb points",ylab="user_time (s)", panel.first=grid())
points(floor(10^logn),times$chol_icpp_elapsed,col='light blue', pch=4)
points(floor(10^logn),times$chol_cpp_user,col='black')
points(floor(10^logn),times$nochol_cpp_elapsed,col='orange', pch=3)
points(floor(10^logn),times$nochol_icpp_elapsed,col='coral', pch=4)
points(floor(10^logn),times$nochol_cpp_user,col='red')
text(100,1.2,"With Cholesky", col= 'black', pos = 2)
text(100,1.2,"+ elapsed", col= 'grey', pos = 4)
text(150,1.2,"x inside C++", col= 'light blue', pos = 4)
text(100,1.0,"Without Cholesky",col = 'red', pos = 2)
text(100,1.0,"+ elapsed", col= 'orange', pos = 4)
text(150,1.0,"x inside C++", col= 'coral', pos = 4)
