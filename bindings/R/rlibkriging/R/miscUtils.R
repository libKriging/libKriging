
f <- function(x) {
    1.0 - 1.0 / 2.0 * (sin(12.0 * x) / (1.0 + x) +
                       2.0 * cos(7.0 * x) * x^5 + 0.7)
}

warnOnDots <- function(dots) {
    warning("Arguments ",
            paste0(names(dots), "=", dots, collapse = ","),
            " are ignored.")
}

## could be a method ???
checkNewX <- function(object, X) {
    
}

## check that 
checkNewTheta <- function(object, theta) {
    
}

## XXXY use 'identical' with formula objects seesm cleaner
formula2regmodel = function(form) {
    if (format(form) == "~1")
        return("constant")
    else if (format(form) == "~.")
        return("linear")
    else if (format(form) == "~.^2")
        return("interactive")
    else stop("Unsupported formula ", form)
}


regmodel2formula = function(regmodel) {
  if (regmodel == "constant")
    return(~1)
  else if (regmodel == "linear")
    return(~.)
  else if (regmodel == "interactive")
    return(~.^2)
  else stop("Unsupported regmodel ",regmodel)
}


## Setup computing parameters used in \code{Kriging} Object

## ****************************************************************************
#' Setup numerical nugget used in Cholesky decompositions
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param x Numerical nugget value.
#' @param ... Ignored
#' 
#' @export
set_num_nugget <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    linalg_set_num_nugget(x)
}


## ****************************************************************************
#' Get numerical nugget used in Cholesky decompositions
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param ... Ignored
#' 
#' @return numerical nugget used.
#' 
#' @export
get_num_nugget <- function(...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    return(linalg_get_num_nugget())
}


## ****************************************************************************
#' Set random ssed used in fit/optim
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param x Seed for mt19937 RNG
#' @param ... Ignored
#' #' 
#' @export
reset_seed <- function(x,...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    random_reset_seed(x)
}