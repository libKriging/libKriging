#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP

#include <bitset>

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

  template <int I, typename T>
  typename converter_trait<T>::type get(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr<I>(msg));
    }
    m_accesses.set(I);
    return converter<T>(m_p[I], parameterStr<I>(msg));
  }

  template <int I, typename T>
  std::optional<typename converter_trait<T>::type> getOptional(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n)
      return std::nullopt;
    m_accesses.set(I);
    return converter<T>(m_p[I], parameterStr<I>(msg));
  }

  template <int I, typename T>
  void set(const T& t, const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr<I>(msg));
    }
    m_accesses.set(I);
    setter<T>(t, m_p[I]);
  }

  template <int I, typename T>
  void setOptional(const T& t, const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      return;
    }
    m_accesses.set(I);
    setter<T>(t, m_p[I]);
  }

  template <int I, typename T>
  T* getObject(const char* msg = nullptr) {
    static_assert(I >= 0);
    if (I >= m_n) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Unavailable ", parameterStr<I>(msg));
    }
    m_accesses.set(I);
    auto ref = get<I, ObjectRef>(msg);
    auto ptr = ObjectCollector::getObject<T>(ref);
    if (ptr == nullptr) {
      throw MxException(LOCATION(), "mLibKriging:missingArg", "Undefined reference object");
    }
    return ptr;
  }

  int count() const { return m_n; }

  template <int I>
  static std::string parameterStr(const char* msg) {
    if (msg) {
      return "parameter " + std::to_string(I) + " '" + msg + "'";
    } else {
      return "parameter " + std::to_string(I);
    }
  }
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXMAPPER_HPP
