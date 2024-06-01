library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)
rlibkriging:::linalg_set_chol_warning(TRUE)
default_rcond_check = rlibkriging:::linalg_chol_rcond_checked()
rlibkriging:::linalg_check_chol_rcond(FALSE)
default_num_nugget = rlibkriging:::linalg_get_num_nugget()
rlibkriging:::linalg_set_num_nugget(1e-15) # lowest nugget to avoid numerical inequalities bw simulates

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
nugget = 0.01
set.seed(1234)
y_o <- f(X_o) + rnorm(n, sd = sqrt(nugget))
points(X_o, y_o,pch=16)

lk <- NuggetKriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), nugget=nugget, sigma2=0.1))


X_n = unique(sort(c(X_o,seq(0,1,,11))))

# lk_nn = Kriging(y = matrix(y_o, ncol = 1),
#               X = matrix(X_o, ncol = 1),
#               kernel = "gauss",
#               regmodel = "linear",
#               optim = "none",
#               #normalize = TRUE,
#               parameters = list(theta = matrix(0.1), sigma2 = 0.1))

## Ckeck consistency bw predict & simulate

lp = NULL
lp = lk$predict(X_n) # libK predict
lines(X_n,lp$mean,col='red')
polygon(c(X_n,rev(X_n)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

ls = NULL
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
y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(nugget))

# new Kriging model from scratch
l2 = NuggetKriging(y = matrix(c(y_o,y_u),ncol=1),
             X = matrix(c(X_o,X_u),ncol=1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              parameters = list(theta = matrix(0.1), nugget=nugget, sigma2 = 0.1))

lu = copy(lk)
lu$update(y_u, X_u, refit=TRUE) # refit=TRUE will update beta (required to match l2)


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

#i_u = c(10,30)
#X_u = X_n[i_u]# c(.4,.6)
#y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(nugget))

X_n = sort(c(X_u-1e-2,X_u+1e-2,X_n)) # add some nugget to avoid degenerate cases

ls = lk$simulate(1000, 123, X_n, with_nugget = TRUE, will_update=TRUE)
#y_u = rs[i_u,1] # force matching 1st sim
lus=NULL
lus = lk$update_simulate(y_u,  X_u)

lu = copy(lk)
lu$update(y_u, X_u, refit=TRUE) # refit=TRUE will update beta (required to match l2)
lsu=NULL
lsu = lu$simulate(1000, 123, X_n, with_nugget = TRUE)

plot(f,xlim=c(0.55,0.65), ylim=c(0.2,0.55))
points(X_o,y_o,pch=16)
for (i in 1:length(X_o)) {
    lines(c(X_o[i],X_o[i]),c(y_o[i]+2*sqrt(nugget),y_o[i]-2*sqrt(nugget)),col='black',lwd=4)
}
points(X_u,y_u,col='red',pch=16)
for (i in 1:length(X_u)) {
    lines(c(X_u[i],X_u[i]),c(y_u[i]+2*sqrt(nugget),y_u[i]-2*sqrt(nugget)),col='red',lwd=4)
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
        X_n[i] + ds$y/20,
        ds$x,
        col=rgb(0,0,0,0.2),border='black')
    polygon(
        X_n[i] + dsu$y/20,
        dsu$x,
        col=rgb(1,0.5,0,0.2),border='orange')
    polygon(
        X_n[i] + dus$y/20,
        dus$x,
        col=rgb(1,0,0,0.2),border='red')
    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
}

for (i in 1:length(X_n)) {
    plot(density(ls[i,]),xlim=range(c(ls[i,],lsu[i,],lus[i,])))
    lines(density(lsu[i,]),col='orange')
    lines(density(lus[i,]),col='red')
    #if (all(abs(X_n[i]-X_u)>1e-2))
    if (sd(lsu[i,])>1e-3 && sd(lus[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("updated,simulated sample follows simulated,updated distribution at x=",X_n[i]," ",
    mean(lus[i,]),",",sd(lus[i,])," != ",mean(lsu[i,]),",",sd(lsu[i,])),
        expect_gt(ks.test(lus[i,],lsu[i,])$p.value, 1e-10)) # just check that it is not clearly wrong
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
y_o <- f(X_o) + rnorm(n, sd = sqrt(nugget))
#points(X_o, y_o)

lkd <- NuggetKriging(y = y_o,
              X = X_o,
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(rep(0.1,d))^2, nugget=nugget, sigma2=0.1^2))

## Predict & simulate

X_n = matrix(runif(100),ncol=d) #seq(0,1,,)

lpd = lkd$predict(X_n) # libK predict
#lines(X_n,lp$mean,col='red')
#polygon(c(X_n,rev(X_n)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)

lsd = lkd$simulate(1000, 123, X_n) # libK simulate
#for (i in 1:100) {
#    lines(X_n,ls[,i],col=rgb(1,0,0,.1),lwd=4)
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
#lp2 = l2$predict(X_n)
#lpu = lu$predict(X_n)
#
#plot(f)
#points(X_o,y_o)
#lines(X_n,lp2$mean,col='red')
#polygon(c(X_n,rev(X_n)),c(lp2$mean+2*lp2$stdev,rev(lp2$mean-2*lp2$stdev)),col=rgb(1,0,0,0.2),border=NA)
#lines(X_n,lpu$mean,col='blue')
#polygon(c(X_n,rev(X_n)),c(lpu$mean+2*lpu$stdev,rev(lpu$mean-2*lpu$stdev)),col=rgb(0,0,1,0.2),border=NA)
#
#ls2 = l2$simulate(100, 123, X_n)
#lsu = lu$simulate(100, 123, X_n)
#for (i in 1:100) {
#    lines(X_n,ls2[,i],col=rgb(1,0,0,.1),lwd=4)
#    lines(X_n,lsu[,i],col=rgb(0,0,1,.1),lwd=4)
#}
#
#for (i in 1:length(X_n)) {
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
y_u = f(X_u) + rnorm(nrow(X_u), sd = sqrt(nugget))

X_n = rbind(X_u+1e-2,X_n) # add some nugget to avoid degenerate cases

#lk = rlibkriging:::load.NuggetKriging("/tmp/lk.json")
lsd = lkd$simulate(1000, 123, X_n, with_nugget=TRUE, will_update=TRUE)
lusd = NULL
lusd = lkd$update_simulate(y_u, X_u)

lud = copy(lkd)
lud$update(matrix(y_u,ncol=1), X_u, refit=TRUE) # refit=TRUE will update beta (required to match l2)
#lu = rlibkriging:::load.NuggetKriging("/tmp/lu.json")
lsud = NULL
lsud = lud$simulate(1000, 123, X_n)

#lk$save("/tmp/lk.json")
#lu$save("/tmp/lu.json")

#plot(f)
#points(X_o,y_o,pch=20)
#points(X_u,y_u,col='red',pch=20)
#for (i in 1:ncol(lus)) {
#    lines(X_n,lus[,i],col=rgb(1,0,0,.1),lwd=4)
#    lines(X_n,lsu[,i],col=rgb(0,0,1,.1),lwd=4)
#}
#
#for (i in 1:1:length(X_n)) {
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
    test_that(desc=paste0("updated,simulated sample follows simulated,updated distribution ",mean(lusd[i,]),",",sd(lusd[i,])," != ",mean(lsud[i,]),",",sd(lsud[i,])),
        expect_gt(ks.test(lusd[i,],lsud[i,])$p.value, 0.01)) # just check that it is not clearly wrong
    }
}

rlibkriging:::linalg_check_chol_rcond(default_rcond_check)
rlibkriging:::linalg_set_num_nugget(default_num_nugget)
