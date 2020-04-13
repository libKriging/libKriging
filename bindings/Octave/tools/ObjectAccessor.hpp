#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP

#include "MxException.hpp"
#include "NonCopyable.hpp"
#include "ObjectCollector.hpp"
#include "RequiresArg.hpp"
#include "mex.h"

struct ObjectRef {};
struct EmptyObject {};

template <typename T, typename... Args>
ObjectCollector::ref_t buildObject(Args... args) {
  return ObjectCollector::registerObject(new T{args...});
}

ObjectCollector::ref_t getObject(const mxArray* obj);

void destroyObject(uint64_t ref);

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTACCESSOR_HPP
