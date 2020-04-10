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
  if (!ObjectCollector::hasInstance()) {
#ifdef MEX_DEBUG    
    mexWarnMsgTxt("ObjectCollector already destroyed");
#endif
    return;  // silent return. Destruction workflow is in progress (ObjectCollector already destroyed)
  }

  auto ref = getObject(obj);
  if (!ObjectCollector::unregisterObject(ref)) {
    throw MxException("mLibKriging:nonExistingRef", "ObjectRef requested to unregister does not exist");
  }
  mxArray* out = mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
  mxSetProperty(obj, 0, "ref", out);
}
