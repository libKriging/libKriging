
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
