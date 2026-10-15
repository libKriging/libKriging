#ifndef RLIBKRIGING_HANDLE_HPP
#define RLIBKRIGING_HANDLE_HPP

// no R / Rcpp include here: each binding file includes RcppArmadillo first
typedef struct SEXPREC* SEXP;

// R serialization (saveRDS / save / PSOCK clusters, #327) of the objects
// holding a C++ model ("Kriging", "NestedKriging").
//
// Their attr(, "object") is an environment holding, as "handle", an ALTREP
// integer of length 1 whose data1 is the XPtr to the C++ model. On
// serialization, the ALTREP Serialized_state method stores the current
// model as its JSON save file; on unserialization, the model is loaded back
// into a new C++ object. The environment is serialized by reference, so the
// object and the closures of its `$` methods keep sharing one C++ model.

/// Wrap a fresh XPtr (of a C++ `type`, "Kriging" or "NestedKriging") as the
/// attr(, "object") of an R object.
SEXP lk_wrap(SEXP xptr, const char* type);

/// The XPtr held by attr(, "object") (also accepts a bare XPtr).
SEXP lk_xptr(SEXP object);

#endif  // RLIBKRIGING_HANDLE_HPP
