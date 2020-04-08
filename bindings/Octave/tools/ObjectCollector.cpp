//
// Created by Pascal Havé on 07/04/2020.
//

#include "ObjectCollector.hpp"

#include <iostream>

void Destroyable::debug(const char* msg, void* t) const {
  std::cout << msg << " " << type << " " << t << std::endl;
}

ObjectCollector::ObjectCollector() {
  std::cout << "Ctor ObjectCollector " << this << std::endl;
}

ObjectCollector::~ObjectCollector() {
  std::cout << "Dtor ObjectCollector " << this << std::endl;
}

ObjectCollector& ObjectCollector::instance() {
  if (!m_instance) {
    m_instance = std::make_unique<ObjectCollector>();
  }
  return *m_instance;
}

void ObjectCollector::unregisterObject(uint64_t ref) {
  auto finder = instance().m_references.find(ref);
  if (finder == instance().m_references.end()) {
    return;  // TODO(pascal) better error management
  }
  instance().m_references.erase(finder);
}

std::unique_ptr<ObjectCollector> ObjectCollector::m_instance;  // NOLINT(fuchsia-statically-constructed-objects)