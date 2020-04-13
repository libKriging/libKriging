#include "ObjectAccessor.hpp"

#include "MxException.hpp"

ObjectCollector::ref_t getObject(const mxArray* obj) {
  if (mxIsEmpty(obj)) {
    throw MxException(LOCATION(), "mLibKriging:emptyObject", "object already contain an empty 'ref' property");
  }

  auto ref = *(static_cast<uint64_t*>(mxGetData(obj)));
  return ref;
}

void destroyObject(uint64_t ref) {
  if (!ObjectCollector::hasInstance()) {
#ifdef MEX_DEBUG
    mexWarnMsgTxt("ObjectCollector already destroyed");
#endif
    return;  // silent return. Destruction workflow is in progress (ObjectCollector already destroyed)
  }

  if (!ObjectCollector::unregisterObject(ref)) {
    throw MxException(LOCATION(), "mLibKriging:nonExistingRef", "ObjectRef requested to unregister does not exist");
  }
}
