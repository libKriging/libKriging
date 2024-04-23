library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
y_o <- f(X_o)
points(X_o, y_o)

lk <- Kriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              parameters = list(theta = matrix(0.1)))

## Predict & simulate

lp = lk$predict(seq(0,1,,21)) # libK predict
lines(seq(0,1,,21),lp$mean,col='red')
polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = lk$simulate(100, 123, seq(0,1,,21)) # libK simulate
for (i in 1:100) {
    lines(seq(0,1,,21),ls[,i],col=rgb(1,0,0,.1),lwd=4)
}

for (i in 1:21) {
    m = lp$mean[i,]
    s = lp$stdev[i,]
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ls[i,] - m,"rnorm",mean=m,sd=s)$p.value < 0.05))
}

## Update

X_u = c(.4,.6)
y_u = f(X_u)

# new Kriging model from scratch
l2 = Kriging(y = matrix(c(y_o,y_u),ncol=1),
             X = matrix(c(X_o,X_u),ncol=1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              parameters = list(theta = matrix(0.1)))

lu = copy(lk)
update(lu, y_u,X_u)

## Update, predict & simulate

lp2 = l2$predict(seq(0,1,,21))
lpu = lu$predict(seq(0,1,,21))

plot(f)
points(X_o,y_o)
lines(seq(0,1,,21),lp2$mean,col='red')
polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lp2$mean+2*lp2$stdev,rev(lp2$mean-2*lp2$stdev)),col=rgb(1,0,0,0.2),border=NA)
lines(seq(0,1,,21),lpu$mean,col='blue')
polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lpu$mean+2*lpu$stdev,rev(lpu$mean-2*lpu$stdev)),col=rgb(0,0,1,0.2),border=NA)

ls2 = l2$simulate(100, 123, seq(0,1,,21))
lsu = lu$simulate(100, 123, seq(0,1,,21))
for (i in 1:100) {
    lines(seq(0,1,,21),ls2[,i],col=rgb(1,0,0,.1),lwd=4)
    lines(seq(0,1,,21),lsu[,i],col=rgb(0,0,1,.1),lwd=4)
}

for (i in 1:21) {
    m2 = lp2$mean[i,]
    s2 = lp2$stdev[i,]
    mu = lpu$mean[i,]
    su = lpu$stdev[i,]
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ls2[i,] - m2,"rnorm",mean=m2,sd=s2)$p.value < 0.05))
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(lsu[i,] - mu,"rnorm",mean=mu,sd=su)$p.value < 0.05))
}



## Update simulate

X_n = seq(0,1,,21)
i_u = c(9,13)
X_u = X_n[i_u]# c(.4,.6)
y_u = f(X_u)

ls = lk$simulate(100, 123, X_n, will_update=TRUE)
#y_u = rs[i_u,1] # force matching 1st sim
lus = lk$update_simulate(y_u, X_u)

lu = copy(lk)
lu$update(y_u, X_u)
lsu = lu$simulate(100, 123, X_n)

for (i in 1:length(X_n)) {
    test_that(desc="updated,simulated sample follows simulated,updated distribution",
        expect_true(ks.test(lus[i,],lsu[i,])$p.value < 0.05))
}
