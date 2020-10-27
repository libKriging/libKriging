## This benchmark is dedicated to compare behavior of OptimLib/Ensmallen and optim (for R) on a given objective function (Rosenbrock)

a=.5
b=50
rosenbrock_fun = function(X) (a-X[,1])^2+b*(X[,2]-X[,1]^2)^2
# min: rosenbrock_fun(cbind(a,a^2))==0

x=seq(0,1,,21)
contour(x,x,matrix(rosenbrock_fun(expand.grid(x,x)),nrow=length(x)),nlevels = 50)
points(a,a^2)

rosenbrock_grad = function(X) cbind(
  -2*(a-X[,1]) + 4*b*(X[,1]^3 - X[,2]*X[,1]),
  2*b*(X[,2] - X[,1]^2)
)
# min: rosenbrock_grad(cbind(a,a^2)) == 0 0
delta=.001
for (x in seq(0,1,,11))
  for (y in seq(0,1,,11)) {
    g = rosenbrock_grad(cbind(x,y))
    arrows(x,y,x-delta*g[,1],y-delta*g[,2],length = .1)
  }




set.seed(1234)
X0 = matrix(runif(2),ncol=2)
#write.csv(X0,"X0.csv")

Xn_optimR = matrix(NA,ncol=2,nrow=0)
Xn_optimC = matrix(NA,ncol=2,nrow=0)

for (ix0 in 1:nrow(X0)) {
  x0 = X0[ix0,]
  
  if (abs(rosenbrock_fun(matrix(x0,ncol=2)) - f_optim(x0))>1E-7) stop("Wrong f eval")
  if (any(abs(rosenbrock_grad(matrix(x0,ncol=2)) - t(grad_optim(x0)))>1E-7)) stop("Wrong g eval")
  
  hist_x = NULL
  last_x = NULL
  f = function(x) {
    n.f <<- n.f+1
    x=matrix(x,ncol=2); 
    if(exists('last_x'))
      if(!is.null(last_x)) 
        lines(x=c(last_x[,1],x[,1]),y=c(last_x[,2],x[,2]),lty=2,col=rgb(0,0,1,.2)); 
    last_x <<- x;
    hist_x <<- rbind(hist_x,x) 
    #points(x,col=rgb(1,0,0,.2)); 
    f_optim(x) #rosenbrock_fun(x)
  }
  g = function(x) {
    n.g <<- n.g+1
    x=matrix(x,ncol=2);
    gr=t(grad_optim(x)) #rosenbrock_grad(x); 
    arrows(x[,1],x[,2],x[,1]-delta*gr[1],x[,2]-delta*gr[2],length = .1,col=rgb(0,0,1,.2)); 
    return(gr)
  }
  
  n.f <<- n.g <<- 0
  o=optim(x0,f,g,method = "L-BFGS-B", control=list(maxit=10)) #,lower = c(0,0),upper=c(1,1))
  points(o$par[1],o$par[2],col='blue',pch='x')
  text(x=x0[1],y=x0[2],paste0(n.f,",",n.g),col='blue')
  Xn_optimR = rbind(Xn_optimR,o$par)
  
  # n.f <<- n.g <<- 0
  X = bench_optim(x0)
  #points(X[,1],X[,2],col=rgb(0,0,1,.2),pch=20)
  lines(X[,],col=rgb(1,0,0,.2),lty=2)
  points(X[nrow(X),1],X[nrow(X),2],col='red',pch='x')
  n.f = nrow(X); n.g = "?"
  text(x=x0[1]+.01,y=x0[2]+.01,paste0(n.f,",",n.g),col='red')
  Xn_optimC  = rbind(Xn_optimC,X[nrow(X),])
}

#write.csv(Xn_optimR,"Xn_optimR.csv")
#write.csv(Xn_optimC,"Xn_optimC.csv")

xn=c(a,a^2)
I = function(X) {
  d = sqrt(rowSums((X-matrix(xn,nrow=nrow(X),ncol=2,byrow=T))^2))
  mean(d^2)+var(d)
}

print(I(Xn_optimR))
#cat(file="Rout.txt",append=T,paste0("\nI_optimR=",I(Xn_optimR),"\n"))
print(I(Xn_optimC))
#cat(file="Rout.txt",append=T,paste0("\nI_OptimC=",I(Xn_optimC),"\n"))



