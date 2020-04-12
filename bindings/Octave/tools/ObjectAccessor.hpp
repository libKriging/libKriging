#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP

#include "MxException.hpp"
#include "NonCopyable.hpp"
#include "ObjectCollector.hpp"
#include "RequiresArg.hpp"
#include "mex.h"

struct ObjectRef {};

template <typename T, typename... Args>
ObjectCollector::ref_t buildObject(mxArray* obj, Args... args) {
  mxArray* objectRef = mxGetProperty(obj, 0, "ref");
  if (objectRef == nullptr) {
    throw MxException(LOCATION(), "mLibKriging:badObject", "object does not contain 'ref' property");
  }

  if (!mxIsEmpty(objectRef)) {
    throw MxException(LOCATION(), "mLibKriging:objectAlreadyBuilt", "object already contain a non empty 'ref' property");
  }

  auto ref = ObjectCollector::registerObject(new T{args...});
  mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *((uint64_t*)mxGetData(out)) = ref;
  mxSetProperty(obj, 0, "ref", out);

  return ref;
}

ObjectCollector::ref_t getObject(const mxArray* obj);

void destroyObject(mxArray* obj);

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP
