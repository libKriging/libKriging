library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

context ("no noise/nugget")

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
noise = 0
sigma2 = 0.01
set.seed(1234)
y_o <- f(X_o) + rnorm(n, sd = sqrt(noise))
points(X_o, y_o,pch=16)

lk_no <- NoiseKriging(y = matrix(y_o, ncol = 1),
              noise = matrix(rep(noise, n), ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              normalize = FALSE,
              parameters = list(theta = matrix(0.1), sigma2 = sigma2))

lk_nu <- NuggetKriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              normalize = FALSE,
              parameters = list(theta = matrix(0.1), nugget=noise, sigma2=sigma2))


lk = Kriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              normalize = FALSE,
              parameters = list(theta = matrix(0.1), sigma2 = sigma2))

test_that("Consistency between Noise(0)Kriging and Kriging", {
    expect_equal(lk$T(), lk_no$T()/sqrt(sigma2))
    expect_equal(lk$M(), lk_no$M()*sqrt(sigma2))
    expect_equal(lk$beta(), lk_no$beta())
    expect_equal(lk$z(), lk_no$z()*sqrt(sigma2))
})

test_that("Consistency between Nugget(0)Kriging and Kriging", {
    expect_equal(lk$T(), lk_nu$T())
    expect_equal(lk$M(), lk_nu$M())
    expect_equal(lk$beta(), lk_nu$beta())
    expect_equal(lk$z(), lk_nu$z())
})

## Simulate

X_n = seq(0,1,,51)

ls = lk$simulate(1000, 123, X_n, will_update=FALSE)
#y_u = rs[i_u,1] # force matching 1st sim

ls_no = lk_no$simulate(1000, 123, X_n, will_update=FALSE)

ls_nu = lk_nu$simulate(1000, 123, X_n, will_update=FALSE)

plot(f)
points(X_o,y_o,pch=16)
for (i in 1:length(X_o)) {
    lines(c(X_o[i],X_o[i]),c(y_o[i]+2*sqrt(noise),y_o[i]-2*sqrt(noise)),col='black',lwd=4)
}
for (i in 1:min(10,ncol(ls))) {
    lines(X_n,ls[,i],col=rgb(0,0,0,.51),lwd=2)
    lines(X_n,ls_nu[,i],col=rgb(1,0.5,0,.51),lwd=4,lty=2)
    lines(X_n,ls_no[,i],col=rgb(1,0,0.5,.51),lwd=6,lty=2)
}

for (i in 1:1:length(X_n)) {
    ds=density(ls[i,])
    ds_nu=density(ls_nu[i,])
    ds_no=density(ls_no[i,])
    polygon(
        X_n[i] + ds$y/20,
        ds$x,
        col=rgb(0,0,0,0.2),border=NA)
    polygon(
        X_n[i] + ds_nu$y/20,
        ds_nu$x,
        col=rgb(0,0,1,0.2),border=NA)
    polygon(
        X_n[i] + ds_no$y/20,
        ds_no$x,
        col=rgb(1,0,0,0.2),border=NA)
    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
}

for (i in 1:length(X_n)) {
    plot(density(ls[i,]),xlim=range(ls[i,], c(ls_no[i,],ls_nu[i,])))
    lines(density(ls_no[i,]),col='orange')
    lines(density(ls_nu[i,]),col='red')
    if (sd(ls_no[i,])>1e-3 && sd(ls_nu[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("updated,simulated sample follows simulated,updated distribution ",
    mean(ls_no[i,]),",",sd(ls_no[i,])," != ",mean(ls_nu[i,]),",",sd(ls_nu[i,])),
        expect_gt(ks.test(ls_no[i,],ls_nu[i,])$p.value, 0.01)) # just check that it is not clearly wrong
}


## Update simulate

#i_u = c(9,13)
#X_u = X_n[i_u]# c(.4,.6)
X_u = c(.4,.6)
y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(noise))
noise_u = rep(noise, length(X_u))

ls = lk$simulate(1000, 123, X_n, will_update=TRUE)
#y_u = rs[i_u,1] # force matching 1st sim
lus=NULL
lus = lk$update_simulate(y_u, X_u)

ls_no = lk_no$simulate(1000, 123, X_n, will_update=TRUE)
lus_no=NULL
lus_no = lk_no$update_simulate(y_u, noise_u, X_u)

ls_nu = lk_nu$simulate(1000, 123, X_n, will_update=TRUE)
lus_nu=NULL
lus_nu = lk_nu$update_simulate(y_u, X_u)

lu = copy(lk)
lu$update(y_u, matrix(X_u,ncol=1), refit=FALSE)
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
for (i in 1:min(10,ncol(lus))) {
    lines(X_n,ls[,i],col=rgb(0,0,0,.1),lwd=2)
    lines(X_n,lus_nu[,i],col=rgb(1,0.5,0,.51),lwd=4,lty=2)
    lines(X_n,lus_no[,i],col=rgb(1,0,0.5,.51),lwd=6,lty=2)
    lines(X_n,lus[,i],col=rgb(1,0,0,.51),lwd=2)
    lines(X_n,lsu[,i],col=rgb(0,0,1,.51),lwd=2)
}


for (i in 1:length(X_n)) {
    plot(density(lus[i,]),xlim=range(lus[i,], c(lus_no[i,],lus_nu[i,])))
    lines(density(lus_no[i,]),col='orange')
    lines(density(lus_nu[i,]),col='red')
    if (sd(lus_no[i,])>1e-3 && sd(lus_nu[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("updated,simulated sample follows simulated,updated distribution ",
    mean(lus_no[i,]),",",sd(lus_no[i,])," != ",mean(lus_nu[i,]),",",sd(lus_nu[i,])),
        expect_gt(ks.test(lus_no[i,],lus_nu[i,])$p.value, 0.01)) # just check that it is not clearly wrong
}

########################################## noise/nugget >0 ##########################################

context ("noise = nugget >0")

f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
noise = 0.01 #0.0000001^2
sigma2 = 0.1
set.seed(1234)
y_o <- f(X_o) + rnorm(n, sd = sqrt(noise))
points(X_o, y_o,pch=16)

lk_no <- NoiseKriging(y = matrix(y_o, ncol = 1),
              noise = matrix(rep(noise, n), ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), sigma2 = sigma2))

lk_nu <- NuggetKriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "gauss",
              regmodel = "linear",
              optim = "none",
              #normalize = TRUE,
              parameters = list(theta = matrix(0.1), nugget=noise, sigma2=sigma2))



test_that("Consistency between Noise(0)Kriging and Kriging", {
    expect_equal(lk_nu$T(), lk_no$T()/sqrt(noise+sigma2))
    expect_equal(lk_nu$M(), lk_no$M()*sqrt(noise+sigma2))
    expect_equal(lk_nu$beta(), lk_no$beta())
    expect_equal(lk_nu$z(), lk_no$z()*sqrt(noise+sigma2))
})


## Simulate

X_n = seq(0,1,,51)

ls_no = NULL
ls_no = lk_no$simulate(1000, 123, X_n, will_update=FALSE)

ls_nu = NULL
ls_nu = lk_nu$simulate(1000, 123, X_n, will_update=FALSE)

plot(f)
points(X_o,y_o,pch=16)
for (i in 1:length(X_o)) {
    lines(c(X_o[i],X_o[i]),c(y_o[i]+2*sqrt(noise),y_o[i]-2*sqrt(noise)),col='black',lwd=4)
}
for (i in 1:min(10,ncol(ls_nu))) {
    lines(X_n,ls_nu[,i],col=rgb(1,0.5,0,.51),lwd=4,lty=2)
    lines(X_n,ls_no[,i],col=rgb(1,0,0.5,.51),lwd=4,lty=2)
}

for (i in 1:1:length(X_n)) {
    ds_nu=density(ls_nu[i,])
    ds_no=density(ls_no[i,])
    polygon(
        X_n[i] + ds_nu$y/20,
        ds_nu$x,
        col=rgb(0,0,1,0.2),border=NA)
    polygon(
        X_n[i] + ds_no$y/20,
        ds_no$x,
        col=rgb(1,0,0,0.2),border=NA)
    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
}

for (i in 1:length(X_n)) {
    plot(density(ls_no[i,]),xlim=range(c(ls_no[i,],ls_nu[i,])))
    lines(density(ls_no[i,]),col='orange')
    lines(density(ls_nu[i,]),col='red')
    if (sd(ls_no[i,])>1e-3 && sd(ls_nu[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("nugget & noise simulated are consistent: ",
    mean(ls_no[i,]),",",sd(ls_no[i,])," != ",mean(ls_nu[i,]),",",sd(ls_nu[i,])),
        expect_gt(ks.test(ls_no[i,],ls_nu[i,])$p.value, 1e-5)) # just check that it is not clearly wrong
}


## Update simulate

#i_u = c(9,13)
#X_u = X_n[i_u]# c(.4,.6)
X_u = c(.4,.6)
y_u = f(X_u) + rnorm(length(X_u), sd = sqrt(noise))
noise_u = rep(noise, length(X_u))

ls_no = lk_no$simulate(1000, 123, X_n, will_update=TRUE)
lus_no=NULL
lus_no = lk_no$update_simulate(y_u, noise_u, X_u)

ls_nu = lk_nu$simulate(1000, 123, X_n, will_update=TRUE)
lus_nu=NULL
lus_nu = lk_nu$update_simulate(y_u, X_u)

plot(f)
points(X_o,y_o,pch=16)
for (i in 1:length(X_o)) {
    lines(c(X_o[i],X_o[i]),c(y_o[i]+2*sqrt(noise),y_o[i]-2*sqrt(noise)),col='black',lwd=4)
}
points(X_u,y_u,col='red',pch=16)
for (i in 1:length(X_u)) {
    lines(c(X_u[i],X_u[i]),c(y_u[i]+2*sqrt(noise),y_u[i]-2*sqrt(noise)),col='red',lwd=4)
}
for (i in 1:min(10,ncol(lus_nu))) {
    lines(X_n,lus_nu[,i],col=rgb(1,0.5,0,.51),lwd=4,lty=2)
    lines(X_n,lus_no[,i],col=rgb(1,0,0.5,.51),lwd=4,lty=2)
}

for (i in 1:1:length(X_n)) {
    dus_nu=density(lus_nu[i,])
    dus_no=density(lus_no[i,])
    polygon(
        X_n[i] + dus_nu$y/100,
        dus_nu$x,
        col=rgb(0,0,1,0.2),border=NA)
    polygon(
        X_n[i] + dus_no$y/100,
        dus_no$x,
        col=rgb(1,0,0,0.2),border=NA)
    #test_that(desc="updated,simulated sample follows simulated,updated distribution",
    #    expect_true(ks.test(lus[i,],lsu[i,])$p.value  > 0.01))
}

for (i in 1:length(X_n)) {
    plot(density(ls_nu[i,]),xlim=range(c(ls_no[i,],ls_nu[i,],lus_no[i,],lus_nu[i,])))
    lines(density(ls_no[i,]),col='blue')
    lines(density(lus_no[i,]),col='orange')
    lines(density(lus_nu[i,]),col='red')
    if (all(abs(X_n[i]-X_o)>0)) # means we are not on design points (where nuggetK is deterministic, while noiseK is not)
    if (sd(lus_no[i,])>1e-3 && sd(lus_nu[i,])>1e-3) # otherwise means that density is ~ dirac, so don't test
    test_that(desc=paste0("nugget & noise updated simulate are consistent at x=",X_n[i]," ",
    mean(lus_no[i,]),",",sd(lus_no[i,])," != ",mean(lus_nu[i,]),",",sd(lus_nu[i,])),
        expect_gt(ks.test(lus_no[i,],lus_nu[i,])$p.value, 1e-7)) # just check that it is not clearly wrong
}
