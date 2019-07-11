# wrapper function to invoke test_binding
test_binding1 <- function() {
  result <- .Call("test_binding1")
  return(result)
}

test_binding2 <- function() {
  result <- .Call("test_binding2")
  return(result)
}
