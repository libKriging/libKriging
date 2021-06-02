# install.packages('Rcpp', repos='http://cran.irsn.fr/')
# install.packages('rlibkriging-version.tgz', repos=NULL)

X <- as.matrix(c(0.0, 0.2, 0.5, 0.8, 1.0))
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
y <- f(X)

library(rlibkriging)
k_R <- Kriging(y, X, "gauss")

print(k_R)

x <- as.matrix(seq(0, 1, , 100))
p <- predict(k_R, x, TRUE, FALSE)

pdf("rplot.pdf") # plot to file
plot(f)
points(X, y)

lines(x, p$mean, col = 'blue')
polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)), border = NA, col = rgb(0, 0, 1, 0.2))

s <- simulate(k_R,nsim = 10, seed = 123, x=x)
plot(f)
points(X,y)
matplot(x,s,col=rgb(0,0,1,0.2),type='l',lty=1,add=T)

Xn <- as.matrix(c(0.3,0.4))
yn <- f(Xn)
print(k_R)
update(k_R, yn, Xn, FALSE)
print(k_R)