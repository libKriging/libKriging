#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LINEARHASHSTORAGE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LINEARHASHSTORAGE_HPP

#include <utility>
#include <vector>

template <typename Key, typename T>
class LinearHashStorage {
 public:
  using value_type = std::pair<const Key, T>;
  struct iterator {
    T& second;
    iterator* operator->() { return this; }
  };

 public:
  LinearHashStorage(size_t reservation = 20) {
    m_keys.reserve(reservation);
    m_data.reserve(reservation);
  }

  std::pair<iterator, bool> emplace(const Key& key, T&& data) {
    for (size_t pos = 0; pos < m_keys.size(); ++pos) {
      if (key == m_keys[pos]) {
        return {iterator{m_data[pos]}, false};
      }
    }
    m_keys.emplace_back(key);
    m_data.emplace_back(std::forward<T>(data));
    return {iterator{m_data.back()}, true};
  }

 private:
  std::vector<Key> m_keys;
  std::vector<T> m_data;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_LINEARHASHSTORAGE_HPP
