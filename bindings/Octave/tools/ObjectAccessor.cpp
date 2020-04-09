#include "ObjectAccessor.hpp"

#include "MxException.hpp"

ObjectCollector::ref_t getObject(const mxArray* obj) {
  mxArray* objectRef = mxGetProperty(obj, 0, "ref");
  if (objectRef == nullptr) {
    throw MxException("mLibKriging:badObject", "object does not contain 'ref' property");
  }

  if (mxIsEmpty(objectRef)) {
    throw MxException("mLibKriging:alreadyBuilt", "object already contain an empty 'ref' property");
  }

  auto ref = *(static_cast<uint64_t*>(mxGetData(objectRef)));
  return ref;
}

void destroyObject(mxArray* obj) {
  auto ref = getObject(obj);
  ObjectCollector::unregisterObject(ref);
  mxArray* out = mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
  mxSetProperty(obj, 0, "ref", out);
}
