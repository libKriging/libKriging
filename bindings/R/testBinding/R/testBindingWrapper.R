# wrapper function to invoke test_binding
test_binding <- function() {
  result <- .Call("test_binding")
  return(result)
}

test_binding2 <- function() {
  result <- .Call("test_binding2")
  return(result)
}
