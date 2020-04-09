//
// Created by Pascal Havé on 08/04/2020.
//

#include "func_new.hpp"

#include <armadillo>
#include <cstring>
#include <iostream>
#include <optional>

#include "../tools/NonCopyable.hpp"
#include "../tools/ObjectCollector.hpp"
#include "libKriging/LinearRegression.hpp"
#include "relative_error.hpp"

template <typename T, typename... Args>
ObjectCollector::ref_t buildObject(mxArray* obj, Args... args) {
  mxArray* objectRef = mxGetProperty(obj, 0, "ref");
  if (objectRef == nullptr) {
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");
  }

  if (!mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain a non empty 'ref' property");
  }

  auto ref = ObjectCollector::registerObject(new T{args...});
  mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *((uint64_t*)mxGetData(out)) = ref;
  mxSetProperty(obj, 0, "ref", out);

  return ref;
}

ObjectCollector::ref_t getObject(const mxArray* obj) {
  mxArray* objectRef = mxGetProperty(obj, 0, "ref");
  if (objectRef == nullptr) {
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");
  }

  if (mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain an empty 'ref' property");
  }

  auto ref = *((uint64_t*)mxGetData(objectRef));
  return ref;
}

void deleteObject(mxArray* obj) {
  auto ref = getObject(obj);
  ObjectCollector::unregisterObject(ref);
  mxArray* out = mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
  mxSetProperty(obj, 0, "ref", out);
}

///////////////////////

template <typename T>
struct trait {
  using type = T;
};

struct ObjectRef {};

template <>
struct trait<ObjectRef> {
  using type = ObjectCollector::ref_t;
};

template <typename T>
auto converter(mxArray*);

template <typename T>
void setter(T&, mxArray*&);

namespace RequiresArg {
struct AtLeast {
} atLeast;
struct Exactly {
} exactly;
struct Autodetect {
} autodetect;
};  // namespace RequiresArg

class MxMapper : public NonCopyable {
 private:
  static constexpr int maxsize = 64;
  static constexpr int autodetected = -1;
  enum class RequiredValue { eAtLeast, eExactly, eAutodetect };

 private:
  const char* m_name;
  const int m_n;
  mxArray** m_p;
  std::bitset<maxsize> m_accesses;

 public:
  MxMapper(const char* name, const int n, mxArray** p, RequiresArg::AtLeast, const unsigned required_args)
      : MxMapper(name, n, p, RequiredValue::eAtLeast, required_args) {}

  MxMapper(const char* name, const int n, mxArray** p, RequiresArg::Exactly, const unsigned required_args)
      : MxMapper(name, n, p, RequiredValue::eExactly, required_args) {}

  MxMapper(const char* name, const int n, mxArray** p)
      : MxMapper(name, n, p, RequiredValue::eAutodetect, autodetected) {}

  MxMapper(const char* name, const int n, mxArray** p, RequiresArg::Autodetect)
      : MxMapper(name, n, p, RequiredValue::eAutodetect, autodetected) {}

 private:
  MxMapper(const char* name, const int n, mxArray** p, const RequiredValue requirement, const int required_args)
      : m_name(name), m_n(n), m_p(p) {
    assert(n < maxsize);
    assert(name != nullptr);
    switch (requirement) {
      case RequiredValue::eAtLeast:
        if (required_args > n) {
          mexErrMsgIdAndTxt("mLibKriging:args", "%s requires at least %d parameters", name, required_args);
        }
        break;
      case RequiredValue::eExactly:
        if (required_args != n) {
          mexErrMsgIdAndTxt("mLibKriging:args", "%s requires exactly %d parameters", name, required_args);
        }
        break;
      case RequiredValue::eAutodetect:
        break;
    }
  }

 public:
  ~MxMapper() {
    for (int i = 0; i < m_n; ++i) {
      if (!m_accesses[i]) {
        mexWarnMsgIdAndTxt("mLibKriging:unusedArgument", "%s argument #%d never used", m_name, i);
      }
    }
  }

  template <int I, typename T>
  typename trait<T>::type get(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      mexErrMsgIdAndTxt("mLibKriging", "Unavailable parameter %s", (msg) ? msg : "");
    }
    m_accesses.set(I);
    return converter<T>(m_p[I]);
  }

  template <int I, typename T>
  std::optional<typename trait<T>::type> getOptional(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n)
      return std::nullopt;
    m_accesses.set(I);
    return converter<T>(m_p[I]);
  }

  template <int I, typename T>
  void set(T& /*t*/, const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      mexErrMsgIdAndTxt("mLibKriging", "Unavailable parameter %s", (msg) ? msg : "");
    }
    m_accesses.set(I);
    mexErrMsgIdAndTxt("mLibKriging:notImplemented", "notImplemented %s", __PRETTY_FUNCTION__);
  }

  template <int I, typename T>
  bool setOptional(T& t, const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      return false;
    }
    m_accesses.set(I);
    setter<T>(t, m_p[I]);
  }

  template <int I, typename T>
  T* getObject(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      mexErrMsgIdAndTxt("mLibKriging", "Unavailable parameter %s", (msg) ? msg : "");
    }
    m_accesses.set(I);
    auto ref = get<I, ObjectRef>(msg);
    auto ptr = ObjectCollector::getObject<LinearRegression>(ref);
    if (ptr == nullptr) {
      mexErrMsgIdAndTxt("mLibKriging", "Undefined reference object");
    }
    return ptr;
  }
};

template <>
auto converter<mxArray*>(mxArray* x) {
  return x;
}

template <>
auto converter<arma::vec>(mxArray* x) {
  if (!mxIsDouble(x) || mxIsComplex(x) || mxGetNumberOfDimensions(x) > 2) {
    mexErrMsgTxt("ERROR");
  }
  const arma::uword nrow = mxGetM(x);
  const arma::uword ncol = mxGetN(x);
  if (ncol > 1) {
    mexErrMsgTxt("ERROR");
  }
  double* data = mxGetPr(x);
  return arma::vec{data, nrow, false, true};
}

template <>
auto converter<arma::mat>(mxArray* x) {
  if (!mxIsDouble(x) || mxIsComplex(x) || mxGetNumberOfDimensions(x) > 2) {
    mexErrMsgTxt("ERROR");
  }
  const arma::uword nrow = mxGetM(x);
  const arma::uword ncol = mxGetN(x);
  double* data = mxGetPr(x);
  return arma::mat{data, nrow, ncol, false, true};
}

template <>
auto converter<ObjectRef>(mxArray* x) {
  return getObject(x);
}

template <>
void setter<arma::vec>(arma::vec& v, mxArray*& x) {
  x = mxCreateNumericMatrix(v.n_rows, v.n_cols, mxDOUBLE_CLASS, mxREAL);
  if (false && v.mem_state == 0 && v.n_elem > arma::arma_config::mat_prealloc) {
    // FIXME hard trick; use internal implementation of arma::~Mat
    arma::access::rw(v.mem_state) = 2;
    mxSetPr(x, v.memptr());
  } else {
    std::memcpy(mxGetPr(x), v.memptr(), sizeof(double) * v.n_rows * v.n_cols);
  }
}

void func_new(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  MxMapper input{"Input", nrhs, (mxArray**)prhs, RequiresArg::exactly, 1};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::exactly, 0};
  buildObject<LinearRegression>(input.get<0, mxArray*>("object reference"));
}

void func_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  MxMapper input{"Input", nrhs, (mxArray**)prhs, RequiresArg::exactly, 1};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::exactly, 0};
  deleteObject(input.get<0, mxArray*>("object reference"));
}

void func_fit(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  MxMapper input{"Input", nrhs, (mxArray**)prhs, RequiresArg::exactly, 3};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::exactly, 0};
  auto* lin_reg = input.getObject<0, LinearRegression>("LinearRegression reference");
  lin_reg->fit(input.get<1, arma::vec>("vector"), input.get<2, arma::mat>("matrix"));
}

void func_predict(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  MxMapper input{"Input", nrhs, (mxArray**)prhs, RequiresArg::exactly, 2};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::atLeast, 1};  // TODO range
  auto* lin_reg = input.getObject<0, LinearRegression>("LinearRegression reference");
  auto [y_pred, stderr_v] = lin_reg->predict(input.get<1, arma::mat>("matrix"));
  output.setOptional<0>(y_pred, "predicted response");
  output.setOptional<1>(stderr_v, "prediction error");
}
