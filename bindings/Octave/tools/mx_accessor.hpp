#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MX_ACCESSOR_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MX_ACCESSOR_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include <cstring>

#include "ObjectAccessor.hpp"
#include "mex.h"

template <typename T>
struct converter_trait {
  using type = T;
};

template <>
struct converter_trait<ObjectRef> {
  using type = ObjectCollector::ref_t;
};

template <typename T>
auto converter(mxArray*, const std::string& parameter);

template <typename T>
void setter(const T&, mxArray*&);

/* Specialization */

template <>
inline auto converter<mxArray*>(mxArray* x, const std::string& parameter) {
  return x;
}

template <>
inline auto converter<std::string>(mxArray* x, const std::string& parameter) {
  if (!mxIsChar(x) || mxGetNumberOfDimensions(x) != 2 || mxGetM(x) != 1 || mxGetM(x) != 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a string");
  }

  char buffer[256];
  if (mxGetString(x, buffer, 256) != 0) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a string");
  }

  return std::string{buffer};
}

template <>
inline auto converter<arma::vec>(mxArray* x, const std::string& parameter) {
  if (!mxIsDouble(x) || mxIsComplex(x) || mxGetNumberOfDimensions(x) > 2) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a vector of double");
  }
  const arma::uword nrow = mxGetM(x);
  const arma::uword ncol = mxGetN(x);
  if (ncol > 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a vector of double");
  }
  double* data = mxGetPr(x);
  return arma::vec{data, nrow, false, true};
}

template <>
inline auto converter<arma::mat>(mxArray* x, const std::string& parameter) {
  if (!mxIsDouble(x) || mxIsComplex(x) || mxGetNumberOfDimensions(x) > 2) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a matrix of double");
  }
  const arma::uword nrow = mxGetM(x);
  const arma::uword ncol = mxGetN(x);
  double* data = mxGetPr(x);
  return arma::mat{data, nrow, ncol, false, true};
}

template <>
inline auto converter<ObjectRef>(mxArray* x, const std::string& parameter) {
  return getObject(x);
}

template <>
inline auto converter<uint64_t>(mxArray* x, const std::string& parameter) {
  if (!mxIsUint64(x) || mxIsComplex(x) || mxGetNumberOfElements(x) != 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not an unsigned 64bits int");
  }
  return *static_cast<uint64_t*>(mxGetData(x));
}

template <>
inline auto converter<int32_t>(mxArray* x, const std::string& parameter) {
  if (!mxIsInt32(x) || mxIsComplex(x) || mxGetNumberOfElements(x) != 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not an unsigned 64bits int");
  }
  return *static_cast<int32_t*>(mxGetData(x));
}

template <>
inline auto converter<bool>(mxArray* x, const std::string& parameter) {
  if (!mxIsLogicalScalar(x) || mxIsComplex(x) || mxGetNumberOfElements(x) != 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not an logical");
  }
  return mxIsLogicalScalarTrue(x);
}

template <>
inline auto converter<double>(mxArray* x, const std::string& parameter) {
  if (!mxIsDouble(x) || mxIsComplex(x) || mxGetNumberOfElements(x) != 1) {
    throw MxException(LOCATION(), "mLibKriging:badType", parameter, " is not a double");
  }
  return *static_cast<double*>(mxGetData(x));
}

template <>
inline void setter<std::string>(const std::string& v, mxArray*& x) {
  x = mxCreateString(v.c_str());
}

template <>
inline void setter<arma::vec>(const arma::vec& v, mxArray*& x) {
  x = mxCreateNumericMatrix(v.n_rows, v.n_cols, mxDOUBLE_CLASS, mxREAL);
  if (false && v.mem_state == 0 && v.n_elem > arma::arma_config::mat_prealloc) {
    // FIXME hard trick; use internal implementation of arma::~Mat
    arma::access::rw(v.mem_state) = 2;
    mxSetPr(x, const_cast<double*>(v.memptr()));
  } else {
    std::memcpy(mxGetPr(x), v.memptr(), sizeof(double) * v.n_rows * v.n_cols);
  }
}

template <>
inline void setter<arma::mat>(const arma::mat& v, mxArray*& x) {
  x = mxCreateNumericMatrix(v.n_rows, v.n_cols, mxDOUBLE_CLASS, mxREAL);
  if (false && v.mem_state == 0 && v.n_elem > arma::arma_config::mat_prealloc) {
    // FIXME hard trick; use internal implementation of arma::~Mat
    arma::access::rw(v.mem_state) = 2;
    mxSetPr(x, const_cast<double*>(v.memptr()));
  } else {
    std::memcpy(mxGetPr(x), v.memptr(), sizeof(double) * v.n_rows * v.n_cols);
  }
}

template <>
inline void setter<uint64_t>(const uint64_t& v, mxArray*& x) {
  x = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<uint64_t*>(mxGetData(x)) = v;
}

template <>
inline void setter<int32_t>(const int32_t& v, mxArray*& x) {
  x = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<int32_t*>(mxGetData(x)) = v;
}

template <>
inline void setter<bool>(const bool& v, mxArray*& x) {
  x = mxCreateLogicalScalar(v);
}

template <>
inline void setter<double>(const double& v, mxArray*& x) {
  x = mxCreateDoubleScalar(v);
}

template <>
inline void setter<EmptyObject>(const EmptyObject& /*v*/, mxArray*& x) {
  x = mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
}

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MX_ACCESSOR_HPP
