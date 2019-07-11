# About NAMESPACE
* http://r-pkgs.had.co.nz/namespace.html

# Rcpp
* http://www.rcpp.org

# Methodology
A typical new function in R binding of libKriging looks like:
```cpp
// [[Rcpp::export]]
OutType featureName(InType1 t1,
                    InType2 t2,
                    InType3 t3 = default_value3) {
    // Body
}
```

If that function uses armadillo types/methods, add the appropriate dependency: 
```cpp
// For using armadillo types
// Also required if lib.h as signature using armadillo types
#include <RcppArmadillo.h>

#include "lib.h"

// [[Rcpp::depends("RcppArmadillo")]] // for armadillo methods
// [[Rcpp::export]]
arma::vec featureName(arma::mat m1,
                      InType2 t2,
                      InType3 t3 = default_value3) {
    // Body with arma:: methods
}
```
