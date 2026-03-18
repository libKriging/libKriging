#include "mx_accessor.hpp"

eMxType get_type(mxArray* x) {
  if (mxIsUint64(x) && !mxIsComplex(x) && mxGetNumberOfElements(x) == 1) {
    return eMxType::Uint64;
  } else if (mxIsInt32(x) && !mxIsComplex(x) && mxGetNumberOfElements(x) == 1) {
    return eMxType::Int32;
  } else if (mxIsLogicalScalar(x) && !mxIsComplex(x) && mxGetNumberOfElements(x) == 1) {
    return eMxType::Logical;
  } else if (mxIsDouble(x) && !mxIsComplex(x) && mxGetNumberOfElements(x) == 1) {
    return eMxType::Scalar;
  } else if (mxIsDouble(x) && !mxIsComplex(x) && mxGetNumberOfDimensions(x) <= 2) {
    return eMxType::Matrix;
  } else if (mxIsClass(x, "string")) {  // Matlab
    return eMxType::String;
  } else {
    // may be string for Octave
    auto* chars = mxArrayToString(x);
    if (chars != nullptr) {
      return eMxType::String;
    } else {
      return eMxType::Unknown;
    }
  }
}

std::ostream& operator<<(std::ostream& o, eMxType type) {
  switch (type) {
    case eMxType::String:
      return o << "string";
    case eMxType::Matrix:
      return o << "matrix";
    case eMxType::Uint64:
      return o << "uint64";
    case eMxType::Int32:
      return o << "int32";
    case eMxType::Logical:
      return o << "logical";
    case eMxType::Scalar:
      return o << "scalar";
    case eMxType::Unknown:
      return o << "unknown";
    default:
      assert(false);
  }
}
