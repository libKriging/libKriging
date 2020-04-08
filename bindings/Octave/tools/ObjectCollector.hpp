//
// Created by Pascal Havé on 07/04/2020.
//

#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTCOLLECTOR_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTCOLLECTOR_HPP

#include <map>
#include <memory>
#include <cassert>

#include "Uncopyable.hpp"

template <typename T>
class DestroyableT;

class Destroyable {
 public:
  Destroyable() = default;
  template <typename T>
  explicit Destroyable(T*) : type(typeid(T).name()) {}
  virtual ~Destroyable() = default;

  
 public:
  template <typename T>
  T* cast() {
    assert(typeid(T).name() == type);
    return static_cast<DestroyableT<T>*>(this)->pointer;
  }
  const char* type = nullptr;

 protected:
  void debug(const char* msg, void* t) const;
};

template <typename T>
class DestroyableT
    : public Destroyable
    , Uncopyable {
 public:
  explicit DestroyableT(T* t) : Destroyable{t}, pointer{t} { debug("Building", pointer); }
  ~DestroyableT() {
    debug("Destroying", pointer);
    delete pointer;
  }
  T* pointer;
};

class ObjectCollector : public Uncopyable {
 public:
  using ref_t = uint64_t;
  
 public:
  ObjectCollector();
  ~ObjectCollector();

 private:
  static ObjectCollector& instance();

 public:
  template <typename T>
  static ref_t registerObject(T* t) {
    ref_t ref = hash(t);
    auto [it, success] = instance().m_references.insert({ref, std::unique_ptr<Destroyable>{}});
    if (success) {
      // Newly inserted
      it->second = std::make_unique<DestroyableT<T>>(t);
    } else {
      // Already inserted
      assert(it->second->cast<T>() == t);
    }
    return ref;
  }

  static void unregisterObject(ref_t ref);

  template <typename T>
  static T* getObject(ref_t ref) {
    auto finder = instance().m_references.find(ref);
    if (finder == instance().m_references.end())
      return nullptr;  // TODO better error management
    T* ptr = finder->second->cast<T>();
    assert(ref == hash(ptr));
    return ptr;
  }

 private:
  std::map<ref_t, std::unique_ptr<Destroyable>> m_references;
  static std::unique_ptr<ObjectCollector> m_instance;

  template <typename T>
  static constexpr ref_t hash(const T* t) {
    return reinterpret_cast<uint64_t>(t);
  }
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OBJECTCOLLECTOR_HPP
