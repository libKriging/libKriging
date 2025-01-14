warnOnDots <- function(dots) {
    warning("Arguments ",
            paste0(names(dots), "=", dots, collapse = ","),
            " are ignored.")
}


## XXXY use 'identical' with formula objects seesm cleaner
formula2regmodel = function(form) {
    if (format(form) == "~0")
        return("none")
    else if (format(form) == "~1")
        return("constant")
    else if (format(form) == "~.")
        return("linear")
    else if (format(form) == "~.^2")
        return("interactive")
    else stop("Unsupported formula ", form)
}


regmodel2formula = function(regmodel) {
  if (regmodel == "none")
    return(~0)
  else if (regmodel == "constant")
    return(~1)
  else if (regmodel == "linear")
    return(~.)
  else if (regmodel == "interactive")
    return(~.^2)
  else stop("Unsupported regmodel ",regmodel)
}

as_numeric_matrix = function(x) {
  if (is.matrix(x) & is.numeric(x))
    return(x)
  if (is.vector(x) & is.numeric(x))
    return(matrix(x,nrow=1))
  if (is.data.frame(x))
    return(data.matrix(x))
  stop(paste0("Data no convertible to numeric matrix: ", typeof(x)))
}
