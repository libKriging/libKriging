# wrapper function to invoke test_binding
manual_demo_binding1 <- function() {
  result <- .Call("demo_binding1")
  return(result)
}

manual_demo_binding2 <- function() {
  result <- .Call("demo_binding2")
  return(result)
}

