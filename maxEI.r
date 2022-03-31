##' Maximization of the Expected Improvement criterion
##' 
##' Given an object of class \code{\link[DiceKriging]{km}} and a set of tuning
##' parameters (\code{lower},\code{upper},\code{parinit}, and \code{control}),
##' \code{max_EI} performs the maximization of the Expected Improvement
##' criterion and delivers the next point to be visited in an EGO-like
##' procedure.
##' 
##' The latter maximization relies on a genetic algorithm using derivatives,
##' \code{\link[rgenoud]{genoud}}. This function plays a central role in the
##' package since it is in constant use in the proposed algorithms. It is
##' important to remark that the information needed about the objective
##' function reduces here to the vector of response values embedded in
##' \code{model} (no call to the objective function or simulator).
##' 
##' The current minimum of the observations can be replaced by an arbitrary
##' value (plugin), which is usefull in particular in noisy frameworks.
##' 
##' 
##' @param model an object of class \code{\link[DiceKriging]{km}} ,
##' @param plugin optional scalar: if provided, it replaces the minimum of the
##' current observations,
##' @param type Kriging type: "SK" or "UK"
##' @param lower vector of lower bounds for the variables to be optimized over,
##' @param upper vector of upper bounds for the variables to be optimized over,
##' @param parinit optional vector of initial values for the variables to be
##' optimized over,
##' @param minimization logical specifying if EI is used in minimiziation or in
##' maximization,
##' @param control optional list of control parameters for optimization.  One
##' can control \code{"pop.size"} (default : [N=3*2^dim for dim<6 and N=32*dim
##' otherwise]), \code{"max.generations"} (12), \code{"wait.generations"} (2)
##' and \code{"BFGSburnin"} (2) of function \code{"genoud"} (see
##' \code{\link[rgenoud]{genoud}}).  Numbers into brackets are the default
##' values
##' @return A list with components:
##' 
##' \item{par}{The best set of parameters found.}
##' 
##' \item{value}{The value of expected improvement at par.}
##' @author David Ginsbourger 
##' 
##' Olivier Roustant
##' 
##' Victor Picheny 
##' @references
##' 
##' D. Ginsbourger (2009), \emph{Multiples metamodeles pour l'approximation et
##' l'optimisation de fonctions numeriques multivariables}, Ph.D. thesis, Ecole
##' Nationale Superieure des Mines de Saint-Etienne, 2009.
##' 
##' D.R. Jones, M. Schonlau, and W.J. Welch (1998), Efficient global
##' optimization of expensive black-box functions, \emph{Journal of Global
##' Optimization}, 13, 455-492.
##' 
##' W.R. Jr. Mebane and J.S. Sekhon (2009), in press, Genetic optimization
##' using derivatives: The rgenoud package for R, \emph{Journal of Statistical
##' Software}.
##' @keywords optimize
##' @examples
##' 
##' set.seed(123)
##' ##########################################################
##' ### "ONE-SHOT" EI-MAXIMIZATION OF THE BRANIN FUNCTION ####
##' ### 	KNOWN AT A 9-POINTS FACTORIAL DESIGN          ####
##' ##########################################################
##' 
##' # a 9-points factorial design, and the corresponding response
##' d <- 2 
##' n <- 9
##' design.fact <- expand.grid(seq(0,1,length=3), seq(0,1,length=3)) 
##' names(design.fact) <- c("x1", "x2")
##' design.fact <- data.frame(design.fact) 
##' names(design.fact) <- c("x1", "x2")
##' response.branin <- apply(design.fact, 1, branin)
##' response.branin <- data.frame(response.branin) 
##' names(response.branin) <- "y" 
##' 
##' # model identification
##' fitted.model1 <- km(~1, design=design.fact, response=response.branin, 
##' covtype="gauss", control=list(pop.size=50,trace=FALSE), parinit=c(0.5, 0.5))
##' 
##' # EGO one step
##' library(rgenoud)
##' lower <- rep(0,d) 
##' upper <- rep(1,d)     # domain for Branin function
##' oEGO <- max_EI(fitted.model1, lower=lower, upper=upper, 
##' control=list(pop.size=20, BFGSburnin=2))
##' print(oEGO)
##' 
##' # graphics
##' n.grid <- 20
##' x.grid <- y.grid <- seq(0,1,length=n.grid)
##' design.grid <- expand.grid(x.grid, y.grid)
##' response.grid <- apply(design.grid, 1, branin)
##' z.grid <- matrix(response.grid, n.grid, n.grid)
##' contour(x.grid,y.grid,z.grid,40)
##' title("Branin Function")
##' points(design.fact[,1], design.fact[,2], pch=17, col="blue")
##' points(oEGO$par[1], oEGO$par[2], pch=19, col="red")
##' 
##' 
##' #############################################################
##' ### "ONE-SHOT" EI-MAXIMIZATION OF THE CAMELBACK FUNCTION ####
##' ###	KNOWN AT A 16-POINTS FACTORIAL DESIGN            ####
##' #############################################################
##' \dontrun{
##' # a 16-points factorial design, and the corresponding response
##' d <- 2 
##' n <- 16
##' design.fact <- expand.grid(seq(0,1,length=4), seq(0,1,length=4)) 
##' names(design.fact)<-c("x1", "x2")
##' design.fact <- data.frame(design.fact) 
##' names(design.fact) <- c("x1", "x2")
##' response.camelback <- apply(design.fact, 1, camelback)
##' response.camelback <- data.frame(response.camelback) 
##' names(response.camelback) <- "y" 
##' 
##' # model identification
##' fitted.model1 <- km(~1, design=design.fact, response=response.camelback, 
##' covtype="gauss", control=list(pop.size=50,trace=FALSE), parinit=c(0.5, 0.5))
##' 
##' # EI maximization
##' library(rgenoud)
##' lower <- rep(0,d) 
##' upper <- rep(1,d)   
##' oEGO <- max_EI(fitted.model1, lower=lower, upper=upper, 
##' control=list(pop.size=20, BFGSburnin=2))
##' print(oEGO)
##' 
##' # graphics
##' n.grid <- 20
##' x.grid <- y.grid <- seq(0,1,length=n.grid)
##' design.grid <- expand.grid(x.grid, y.grid)
##' response.grid <- apply(design.grid, 1, camelback)
##' z.grid <- matrix(response.grid, n.grid, n.grid)
##' contour(x.grid,y.grid,z.grid,40)
##' title("Camelback Function")
##' points(design.fact[,1], design.fact[,2], pch=17, col="blue")
##' points(oEGO$par[1], oEGO$par[2], pch=19, col="red")
##' }
##' 
##' ####################################################################
##' ### "ONE-SHOT" EI-MAXIMIZATION OF THE GOLDSTEIN-PRICE FUNCTION #####
##' ### 	     KNOWN AT A 9-POINTS FACTORIAL DESIGN              #####
##' ####################################################################
##' 
##' \dontrun{
##' # a 9-points factorial design, and the corresponding response
##' d <- 2 
##' n <- 9
##' design.fact <- expand.grid(seq(0,1,length=3), seq(0,1,length=3)) 
##' names(design.fact)<-c("x1", "x2")
##' design.fact <- data.frame(design.fact) 
##' names(design.fact)<-c("x1", "x2")
##' response.goldsteinPrice <- apply(design.fact, 1, goldsteinPrice)
##' response.goldsteinPrice <- data.frame(response.goldsteinPrice) 
##' names(response.goldsteinPrice) <- "y" 
##' 
##' # model identification
##' fitted.model1 <- km(~1, design=design.fact, response=response.goldsteinPrice, 
##' covtype="gauss", control=list(pop.size=50, max.generations=50, 
##' wait.generations=5, BFGSburnin=10, trace=FALSE), parinit=c(0.5, 0.5), optim.method="gen")
##' 
##' # EI maximization
##' library(rgenoud)
##' lower <- rep(0,d); upper <- rep(1,d);     # domain for Branin function
##' oEGO <- max_EI(fitted.model1, lower=lower, upper=upper, control
##' =list(pop.size=50, max.generations=50, wait.generations=5, BFGSburnin=10))
##' print(oEGO)
##' 
##' # graphics
##' n.grid <- 20
##' x.grid <- y.grid <- seq(0,1,length=n.grid)
##' design.grid <- expand.grid(x.grid, y.grid)
##' response.grid <- apply(design.grid, 1, goldsteinPrice)
##' z.grid <- matrix(response.grid, n.grid, n.grid)
##' contour(x.grid,y.grid,z.grid,40)
##' title("Goldstein-Price Function")
##' points(design.fact[,1], design.fact[,2], pch=17, col="blue")
##' points(oEGO$par[1], oEGO$par[2], pch=19, col="red")
##' }
##' 
##' @export max_EI
max_EI <-function(model, plugin=NULL, type = "UK", lower, upper, parinit=NULL, minimization = TRUE, control=NULL) {

  if (is.null(plugin)){ plugin <- min(model@y) }
  
	EI.envir <- new.env()	
	environment(EI) <- environment(EI.grad) <- EI.envir 
	gr = EI.grad
  
	d <- ncol(model@X)

	if (is.null(control$print.level)) control$print.level <- 1
	if(d<=6) N <- 3*2^d else N <- 32*d 
	if (is.null(control$BFGSmaxit)) control$BFGSmaxit <- N
	if (is.null(control$pop.size))  control$pop.size <- N
	if (is.null(control$solution.tolerance))  control$solution.tolerance <- 1e-21
	if (is.null(control$max.generations))  control$max.generations <- 12
	if (is.null(control$wait.generations))  control$wait.generations <- 2
	if (is.null(control$BFGSburnin)) control$BFGSburnin <- 2
	if (is.null(parinit))  parinit <- lower + runif(d)*(upper-lower)
     
	domaine <- cbind(lower, upper)

	o <- genoud(EI, nvars=d, max=TRUE,
	            pop.size=control$pop.size, max.generations=control$max.generations, wait.generations=control$wait.generations,
	            hard.generation.limit=TRUE, starting.values=parinit, MemoryMatrix=TRUE, 
	            Domains=domaine, default.domains=10, solution.tolerance=control$solution.tolerance,
	            gr=gr, boundary.enforcement=2, lexical=FALSE, gradient.check=FALSE, BFGS=TRUE,
	            data.type.int=FALSE, hessian=FALSE, unif.seed=floor(runif(1,max=10000)), int.seed=floor(runif(1,max=10000)), 
	            print.level=control$print.level,  
	            share.type=0, instance.number=0, output.path="stdout", output.append=FALSE, project.path=NULL,
	            P1=50, P2=50, P3=50, P4=50, P5=50, P6=50, P7=50, P8=50, P9=0, P9mix=NULL, 
	            BFGSburnin=control$BFGSburnin, BFGSfn=NULL, BFGShelp=NULL, control=list("maxit"=control$BFGSmaxit), 
	            cluster=FALSE, balance=FALSE, debug=FALSE,
	            model=model, plugin=plugin, type=type,minimization = minimization, envir=EI.envir
	)
  
# 	o <- genoud(EI, nvars=d, max=TRUE, 
# 		pop.size=control$pop.size,
# 		max.generations=control$max.generations, 
# 		wait.generations=control$wait.generations,
#     hard.generation.limit=TRUE, starting.values=parinit, MemoryMatrix=TRUE, 
#     Domains=domaine, default.domains=10, solution.tolerance=0.00001, 
#     gr=gr, boundary.enforcement=2, lexical=FALSE, gradient.check=TRUE, BFGS=TRUE,
#     data.type.int=FALSE, hessian=FALSE, unif.seed=floor(runif(1,max=10000)), int.seed=floor(runif(1,max=10000)),
# 	  print.level=control$print.level, 
#     share.type=0, instance.number=0, output.path="stdout", output.append=FALSE, project.path=NULL,
#     P1=50, P2=50, P3=50, P4=50, P5=50, P6=50, P7=50, P8=50, P9=0,	P9mix=NULL, 
#     BFGSburnin=control$BFGSburnin,BFGSfn=NULL, BFGShelp=NULL,
#     cluster=FALSE, balance=FALSE, debug=TRUE, 
#     model=model, plugin=plugin, type=type, envir=EI.envir 
# 		)
                            
  o$par <- t(as.matrix(o$par))
	colnames(o$par) <- colnames(model@X)
	o$value <- as.matrix(o$value)
	colnames(o$value) <- "EI"  
	return(list(par=o$par, value=o$value)) 
}