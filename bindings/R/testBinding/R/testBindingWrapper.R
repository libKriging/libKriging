# wrapper function to invoke test_binding
test_binding_1 <- function() {
  result <- .Call("test_binding")
  return(result)
}

test_binding_2 <- function() {
  result <- .Call("test_binding2")
  return(result)
}
