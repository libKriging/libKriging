#include "ObjectCollector.hpp"

#include <iostream>

void Destroyable::debug(const char* msg, void* t) const {
#ifndef NDEBUG
  std::cout << msg << " " << type << " " << t << std::endl;
#endif
}

ObjectCollector::ObjectCollector() {
#ifndef NDEBUG
  std::cout << "Ctor ObjectCollector " << this << std::endl;
#endif
}

ObjectCollector::~ObjectCollector() {
#ifndef NDEBUG
  std::cout << "Dtor ObjectCollector " << this << std::endl;
#endif
}

ObjectCollector& ObjectCollector::instance() {
  if (!m_instance) {
    m_instance = std::make_unique<ObjectCollector>();
  }
  return *m_instance;
}

bool ObjectCollector::unregisterObject(uint64_t ref) {
  auto finder = instance().m_references.find(ref);
  if (finder == instance().m_references.end()) {
    return false;  // TODO(pascal) better error management
  }
  instance().m_references.erase(finder);
  return true;
}

std::unique_ptr<ObjectCollector> ObjectCollector::m_instance;  // NOLINT(fuchsia-statically-constructed-objects)