library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
noise = 0.01
set.seed(1234)
y_o <- f(X_o) + rnorm(n, sd = sqrt(noise))
points(X_o, y_o,pch=16)

lk <- NoiseKriging(y = matrix(y_o, ncol = 1),
              noise = matrix(rep(noise, n), ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), sigma2 = 0.1))

X_n = unique(sort(c(X_o,seq(0,1,,51))))

lk_nn = Kriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), sigma2 = 0.1))

## Ckeck consistency bw predict & simulate

lp = lk$predict(X_n) # libK predict
lines(X_n,lp$mean,col='red')
polygon(c(X_n,rev(X_n)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = lk$simulate(1000, 123, X_n) # libK simulate
for (i in 1:min(100,ncol(ls))) {
    lines(X_n,ls[,i],col=rgb(1,0,0,.1),lwd=4)
}

for (i in 1:length(X_n)) {
    if (lp$stdev[i,] > 1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(ls[i,], "pnorm", mean = lp$mean[i,],sd = lp$stdev[i,])$p.value > 0.01))
}




## Check consistency when update

X_u = c(.4,.6)
y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(noise))
noise_u = rep(noise, length(X_u))

# new Kriging model from scratch
l2 = NoiseKriging(y = matrix(c(y_o,y_u),ncol=1),
                noise = matrix(c(rep(noise,n),noise_u),ncol=1),
             X = matrix(c(X_o,X_u),ncol=1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              parameters = list(theta = matrix(0.1), sigma2 = 0.1))

lu = copy(lk)
lu$update(y_u, noise_u, X_u)


## Update, predict & simulate

lp2 = l2$predict(X_n)
lpu = lu$predict(X_n)

plot(f)
points(X_o,y_o)
lines(X_n,lp2$mean,col='red')
polygon(c(X_n,rev(X_n)),c(lp2$mean+2*lp2$stdev,rev(lp2$mean-2*lp2$stdev)),col=rgb(1,0,0,0.2),border=NA)
lines(X_n,lpu$mean,col='blue')
polygon(c(X_n,rev(X_n)),c(lpu$mean+2*lpu$stdev,rev(lpu$mean-2*lpu$stdev)),col=rgb(0,0,1,0.2),border=NA)

ls2 = l2$simulate(1000, 123, X_n)
lsu = lu$simulate(1000, 123, X_n)
for (i in 1:100) {
    lines(X_n,ls2[,i],col=rgb(1,0,0,.1),lwd=4)
    lines(X_n,lsu[,i],col=rgb(0,0,1,.1),lwd=4)
}

for (i in 1:length(X_n)) {
    #test_that(desc="simulate sample follows predictive distribution",
    #    expect_true(ks.test(ls2[i,],lsu[i,])$p.value  > 0.01))

    # random gen is the same so we expect strict equality of samples !
    test_that(desc="simulate sample are the same",
        expect_equal(ls2[i,],lsu[i,],tolerance=1e-5))
}



## Update simulate

i_u = c(9,13)
X_u = X_n[i_u]# c(.4,.6)
y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(noise))
noise_u = rep(noise, length(X_u))

X_n = sort(c(X_u+1e-3,X_n)) # add some nugget to avoid degenerate cases

ls = lk$simulate(1000, 123, X_n, will_update=TRUE)
#y_u = rs[i_u,1] # force matching 1st sim
lus=NULL
lus = lk$update_simulate(y_u, noise_u, X_u)

#ls_nn = lk_nn$simulate(10, 123, X_n, will_update=TRUE)
#lus_nn=NULL
#lus_nn = lk_nn$update_simulate(y_u, X_u)

lu = copy(lk)
lu$update(y_u, noise_u, matrix(X_u,ncol=1), refit=FALSE)
lsu=NULL
lsu = lu$simulate(1000, 123, X_n)

plot(f)
points(X_o,y_o,pch=16)
for (i in 1:length(X_o)) {
    lines(c(X_o[i],X_o[i]),c(y_o[i]+2*sqrt(noise),y_o[i]-2*sqrt(noise)),col='black',lwd=4)
}
points(X_u,y_u,col='red',pch=16)
for (i in 1:length(X_u)) {
    lines(c(X_u[i],X_u[i]),c(y_u[i]+2*sqrt(noise),y_u[i]-2*sqrt(noise)),col='red',lwd=4)
}
for (j in 1:min(100,ncol(lus))) {
    lines(X_n,ls[,j],col=rgb(0,0,0,.1),lwd=4)
    lines(X_n,lus[,j],col=rgb(1,0,0,.1),lwd=4)
    lines(X_n,lsu[,j],col=rgb(1,0.5,0,.1),lwd=4)
}

for (i in 1:length(X_n)) {
    ds=density(ls[i,])
    dsu=density(lsu[i,])
    dus=density(lus[i,])
    polygon(
        X_n[i] + ds$y/50,
        ds$x,
        col=rgb(0,0,0,0.2),border=NA)
    polygon(
        X_n[i] + dsu$y/50,
        dsu$x,
        col=rgb(1,0.5,0,0.2),border=NA)
    polygon(
        X_n[i] + dus$y/50,
        dus$x,
        col=rgb(1,0,0,0.2),border=NA)
    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
}

for (i in 1:length(X_n)) {
    plot(density(ls[i,]),xlim=range(c(ls[i,],lsu[i,],lus[i,])))
    lines(density(lsu[i,]),col='orange')
    lines(density(lus[i,]),col='red')
    if (sd(lsu[i,])>1e-3 && sd(lus[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="updated,simulated sample follows simulated,updated distribution",
        expect_true(ks.test(lus[i,],lsu[i,])$p.value > 0.01))
}








############################## 2D ########################################

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 100
d <- 2

set.seed(1234)
X_o <- matrix(runif(n*d),ncol=d) #seq(from = 0, to = 1, length.out = n)
y_o <- f(X_o) + rnorm(n, sd = sqrt(noise))
#points(X_o, y_o)

lkd <- NoiseKriging(y = y_o,
              noise = rep(noise,n),
              X = X_o,
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(rep(0.1,d)), sigma2 = 0.1^2))

## Predict & simulate

X_n = matrix(runif(100),ncol=d) #seq(0,1,,)

lpd = lkd$predict(X_n) # libK predict
#lines(seq(0,1,,21),lp$mean,col='red')
#polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

lsd = lkd$simulate(1000, 123, X_n) # libK simulate
#for (i in 1:100) {
#    lines(seq(0,1,,21),ls[,i],col=rgb(1,0,0,.1),lwd=4)
#}

for (i in 1:nrow(X_n)) {
    m = lpd$mean[i,]
    s = lpd$stdev[i,]
    if (s > 1e-2) # otherwise means that density is ~ dirac, so don't test
    test_that(desc="simulate sample follows predictive distribution",
        expect_true(ks.test(lsd[i,],"pnorm",mean=m,sd=s)$p.value > 0.01))
}

### Update
#
#X_u = c(.4,.6)
#y_u = f(X_u)
#
## new Kriging model from scratch
#l2 = Kriging(y = matrix(c(y_o,y_u),ncol=1),
#             X = matrix(c(X_o,X_u),ncol=1),
#              kernel = "gauss",
#              regmodel = "linear",
#              optim = "none",
#              parameters = list(theta = matrix(0.1)))
#
#lu = copy(lk)
#update(lu, y_u,X_u)
#
### Update, predict & simulate
#
#lp2 = l2$predict(seq(0,1,,21))
#lpu = lu$predict(seq(0,1,,21))
#
#plot(f)
#points(X_o,y_o)
#lines(seq(0,1,,21),lp2$mean,col='red')
#polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lp2$mean+2*lp2$stdev,rev(lp2$mean-2*lp2$stdev)),col=rgb(1,0,0,0.2),border=NA)
#lines(seq(0,1,,21),lpu$mean,col='blue')
#polygon(c(seq(0,1,,21),rev(seq(0,1,,21))),c(lpu$mean+2*lpu$stdev,rev(lpu$mean-2*lpu$stdev)),col=rgb(0,0,1,0.2),border=NA)
#
#ls2 = l2$simulate(100, 123, seq(0,1,,21))
#lsu = lu$simulate(100, 123, seq(0,1,,21))
#for (i in 1:100) {
#    lines(seq(0,1,,21),ls2[,i],col=rgb(1,0,0,.1),lwd=4)
#    lines(seq(0,1,,21),lsu[,i],col=rgb(0,0,1,.1),lwd=4)
#}
#
#for (i in 1:21) {
#    m2 = lp2$mean[i,]
#    s2 = lp2$stdev[i,]
#    mu = lpu$mean[i,]
#    su = lpu$stdev[i,]
#    test_that(desc="simulate sample follows predictive distribution",
#        expect_true(ks.test(ls2[i,] - m2,"pnorm",mean=m2,sd=s2)$p.value  > 0.01))
#    test_that(desc="simulate sample follows predictive distribution",
#        expect_true(ks.test(lsu[i,] - mu,"pnorm",mean=mu,sd=su)$p.value  > 0.01))
#}
#
#
#
## Update simulate

i_u = c(9,13,25,43,42,35,24)
X_u = X_n[i_u,,drop=FALSE]# c(.4,.6)
y_u = f(X_u) + rnorm(nrow(X_u), sd = sqrt(noise))

X_n = rbind(X_u+1e-2,X_n) # add some noise to avoid degenerate cases

#lk = rlibkriging:::load.NoiseKriging("/tmp/lk.json")
lsd = lkd$simulate(1000, 123, X_n, will_update=TRUE)
lusd = NULL
lusd = lkd$update_simulate(y_u, rep(noise,length(y_u)), X_u)

lud = copy(lkd)
lud$update(matrix(y_u,ncol=1), rep(noise,length(y_u)), X_u, refit=FALSE)
#lu = rlibkriging:::load.NoiseKriging("/tmp/lu.json")
lsud = NULL
lsud = lud$simulate(1000, 123, X_n)

#lk$save("/tmp/lk.json")
#lu$save("/tmp/lu.json")

#plot(f)
#points(X_o,y_o,pch=20)
#points(X_u,y_u,col='red',pch=20)
#for (i in 1:ncol(lus)) {
#    lines(seq(0,1,,21),lus[,i],col=rgb(1,0,0,.1),lwd=4)
#    lines(seq(0,1,,21),lsu[,i],col=rgb(0,0,1,.1),lwd=4)
#}
#
#for (i in 1:length(X_n)) {
#    dsu=density(lsu[i,])
#    dus=density(lus[i,])
#    polygon(
#        X_n[i] + dsu$y/100,
#        dsu$x,
#        col=rgb(0,0,1,0.2),border=NA)
#    polygon(
#        X_n[i] + dus$y/100,
#        dus$x,
#        col=rgb(1,0,0,0.2),border=NA)
#    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
#    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
#}

for (i in 1:nrow(X_n)) {
    plot(density(lsd[i,]),xlim=c(0,1))
    lines(density(lsud[i,]),col='orange')
    lines(density(lusd[i,]),col='red')
    if (sd(lsud[i,])>1e-3 && sd(lusd[i,])>1e-3) {# otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("updated,simulated sample follows simulated,updated distribution ",sd(lsud[i,]),",",sd(lusd[i,])),
        expect_gt(ks.test(lusd[i,],lsud[i,])$p.value, 0.01)) # just check that it is not clearly wrong
    }
}
