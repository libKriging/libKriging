// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include <R_ext/Altrep.h>

#include "libKriging/Kriging.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/NestedKriging.hpp"
#include "libKriging/WarpKriging.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "handle.hpp"
#include "retrofit_utils.hpp"

// See handle.hpp. The ALTREP data1 is the XPtr, data2 a list (type, 0L): the
// C++ type name and the materialized integer value of the handle.

static R_altrep_class_t lk_handle_class;

static SEXP make_handle(SEXP xptr, const char* type) {
  SEXP data2 = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(data2, 0, Rf_mkString(type));
  SET_VECTOR_ELT(data2, 1, Rf_ScalarInteger(0));
  SEXP h = R_new_altrep(lk_handle_class, xptr, data2);
  UNPROTECT(1);
  return h;
}

static std::string handle_type(SEXP h) {
  return CHAR(STRING_ELT(VECTOR_ELT(R_altrep_data2(h), 0), 0));
}

static std::string temp_json_file() {
  Rcpp::Function tempfile("tempfile");
  return Rcpp::as<std::string>(tempfile(Rcpp::Named("fileext") = ".json"));
}

// ---- ALTREP methods ----

static R_xlen_t handle_Length(SEXP) {
  return 1;
}

static int handle_Elt(SEXP, R_xlen_t) {
  return 0;
}

static void* handle_Dataptr(SEXP h, Rboolean) {
  return INTEGER(VECTOR_ELT(R_altrep_data2(h), 1));
}

static Rboolean handle_Inspect(SEXP h, int, int, int, void (*)(SEXP, int, int, int)) {
  Rprintf(" rlibkriging handle (%s)\n", handle_type(h).c_str());
  return TRUE;
}

// state = list(type, raw JSON save file); NULL (serialized as a plain
// integer, unserialized as an invalid handle) if the model cannot be saved
static SEXP handle_Serialized_state(SEXP h) {
  SEXP xptr = R_altrep_data1(h);
  if (TYPEOF(xptr) != EXTPTRSXP || R_ExternalPtrAddr(xptr) == nullptr)
    return R_NilValue;
  const std::string type = handle_type(h);
  std::string filename;
  try {
    filename = temp_json_file();
    if (type == "Kriging")
      static_cast<Kriging*>(R_ExternalPtrAddr(xptr))->save(filename);
    else if (type == "NestedKriging")
      static_cast<NestedKriging*>(R_ExternalPtrAddr(xptr))->save(filename);
    else if (type == "WarpKriging")
      static_cast<libKriging::WarpKriging*>(R_ExternalPtrAddr(xptr))->save(filename);
    else if (type == "MLPKriging")
      static_cast<libKriging::MLPKriging*>(R_ExternalPtrAddr(xptr))->save(filename);
    else
      return R_NilValue;
  } catch (const std::exception& e) {
    if (!filename.empty())
      std::remove(filename.c_str());
    Rf_warning("rlibkriging: %s model not serialized (%s)", type.c_str(), e.what());  // no longjmp
    return R_NilValue;
  }
  std::ifstream in(filename, std::ios::binary);
  const std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();
  std::remove(filename.c_str());

  SEXP state = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(state, 0, Rf_mkString(type.c_str()));
  SEXP raw = Rf_allocVector(RAWSXP, static_cast<R_xlen_t>(json.size()));
  SET_VECTOR_ELT(state, 1, raw);
  std::copy(json.begin(), json.end(), reinterpret_cast<char*>(RAW(raw)));
  UNPROTECT(1);
  return state;
}

static SEXP handle_Unserialize(SEXP, SEXP state) {
  // no Rf_error (longjmp) while C++ objects are alive: errors are reported
  // after this block, from a plain char buffer
  static char err[1024];
  err[0] = '\0';
  SEXP xptr = R_NilValue;
  {
    const std::string type = CHAR(STRING_ELT(VECTOR_ELT(state, 0), 0));
    SEXP raw = VECTOR_ELT(state, 1);
    try {
      const std::string filename = temp_json_file();
      {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(RAW(raw)), XLENGTH(raw));
      }
      try {
        if (type == "Kriging") {
          Kriging ok = Kriging::load(filename);
          xptr = Rcpp::XPtr<Kriging>(new Kriging(ok, ExplicitCopySpecifier{}));
        } else if (type == "NestedKriging") {
          xptr = Rcpp::XPtr<NestedKriging>(new NestedKriging(NestedKriging::load(filename)));
        } else if (type == "WarpKriging") {
          xptr = Rcpp::XPtr<libKriging::WarpKriging>(
              new libKriging::WarpKriging(libKriging::WarpKriging::load(filename)));
        } else if (type == "MLPKriging") {
          xptr = Rcpp::XPtr<libKriging::MLPKriging>(new libKriging::MLPKriging(libKriging::MLPKriging::load(filename)));
        } else {
          std::snprintf(err, sizeof(err), "unknown model type %s", type.c_str());
        }
      } catch (...) {
        std::remove(filename.c_str());
        throw;
      }
      std::remove(filename.c_str());
    } catch (const std::exception& e) {
      std::snprintf(err, sizeof(err), "cannot unserialize %s model (%s)", type.c_str(), e.what());
    }
    if (err[0] == '\0') {
      PROTECT(xptr);
      xptr = make_handle(xptr, type.c_str());
      UNPROTECT(1);
    }
  }
  if (err[0] != '\0')
    Rf_error("rlibkriging: %s", err);
  return xptr;
}

// [[Rcpp::init]]
void lk_handle_init(DllInfo* dll) {
  lk_handle_class = R_make_altinteger_class("lk_handle", "rlibkriging", dll);
  R_set_altrep_Length_method(lk_handle_class, handle_Length);
  R_set_altrep_Inspect_method(lk_handle_class, handle_Inspect);
  R_set_altrep_Serialized_state_method(lk_handle_class, handle_Serialized_state);
  R_set_altrep_Unserialize_method(lk_handle_class, handle_Unserialize);
  R_set_altvec_Dataptr_method(lk_handle_class, handle_Dataptr);
  R_set_altinteger_Elt_method(lk_handle_class, handle_Elt);
}

// ---- API (handle.hpp) ----

SEXP lk_wrap(SEXP xptr, const char* type) {
  SEXP env = PROTECT(R_NewEnv(R_EmptyEnv, FALSE, 0));
  SEXP h = PROTECT(make_handle(xptr, type));
  Rf_defineVar(Rf_install("handle"), h, env);
  UNPROTECT(2);
  return env;
}

SEXP lk_xptr(SEXP object) {
  if (TYPEOF(object) == EXTPTRSXP)
    return object;
  if (TYPEOF(object) == ENVSXP) {
    SEXP h = Rcpp::Environment(object).get("handle");  // (Rf_findVarInFrame is not API in R >= 4.5)
    if (ALTREP(h) && R_altrep_inherits(h, lk_handle_class))
      return R_altrep_data1(h);
  }
  Rcpp::stop("invalid libKriging object: the C++ model is missing (an unfitted model, or one serialized "
             "by an older rlibkriging, cannot be restored by readRDS/load; use $save()/load.Kriging())");
}
