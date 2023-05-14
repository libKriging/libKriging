#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP

#include <bitset>
#include <optional>

#include "ObjectCollector.hpp"
#include "RequiresArg.hpp"
#include "mx_accessor.hpp"

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
  MxMapper(const char* name,
           const int n,
           mxArray** p,
           const RequiresArg::Requirement& requirement = RequiresArg::Autodetect{});

 public:
  ~MxMapper();

  eMxType getType(const int I, const char* msg = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr(I, msg));
    }
    m_accesses.set(I);
    return get_type(m_p[I]);
  }

  template <typename T>
  typename converter_trait<T>::type get(const int I, const char* msg = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr(I, msg));
    }
    m_accesses.set(I);
    return converter<T>(m_p[I], parameterStr(I, msg));
  }

  template <typename T>
  std::optional<typename converter_trait<T>::type> getOptional(int I, const char* msg = nullptr) {
    assert(I >= 0);
    if (I >= m_n)
      return std::nullopt;
    m_accesses.set(I);
    return converter<T>(m_p[I], parameterStr(I, msg));
  }

  template <typename T>
  void set(int I, const T& t, const char* msg = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr(I, msg));
    }
    m_accesses.set(I);
    setter<T>(t, m_p[I]);
  }

  template <typename T>
  void setOptional(int I, const T& t, const char* /*msg*/ = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      return;
    }
    m_accesses.set(I);
    setter<T>(t, m_p[I]);
  }

  template <typename T>
  T* getObjectFromRef(int I, const char* msg = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr(I, msg));
    }
    m_accesses.set(I);
    auto ref = get<ObjectRef>(I, msg);
    auto ptr = ObjectCollector::getObject<T>(ref);
    if (ptr == nullptr) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Undefined reference object");
    }
    return ptr;
  }

  template <typename T>
  std::optional<T*> getOptionalObject(int I, const char* = nullptr) {
    assert(I >= 0);
    if (I >= m_n) {
      return std::nullopt;
    }
    m_accesses.set(I);
    // mxGetClassName(m_p[I]); // if you need more info
    mxArray* ref_array = mxGetProperty(m_p[I], 0, "ref");  // by convention how we build object
    ObjectCollector::ref_t ref = getObject(ref_array);
    auto ptr = ObjectCollector::getObject<T>(ref);
    if (ptr == nullptr) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Undefined reference object");
    }
    return std::make_optional<T*>(ptr);
  }

  [[nodiscard]] int count() const { return m_n; }

  static std::string parameterStr(int I, const char* msg) {
    if (msg != nullptr) {
      return "parameter " + std::to_string(I) + " '" + msg + "'";
    } else {
      return "parameter " + std::to_string(I);
    }
  }
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP
