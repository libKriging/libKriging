#include "libKriging/CacheFunction.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

/* ----------------------------------------------------------------------------------------------------------------- */

CacheFunctionCommon::~CacheFunctionCommon() {
  std::cout << stat() << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------------- */

auto CacheFunctionCommon::stat() -> CacheStat {
  std::vector<uint32_t> hits;
  const auto cache_size = m_cache_hit.size();
  hits.reserve(cache_size);
  for (auto [_, e] : m_cache_hit) {
    hits.push_back(e);
  }
  const auto min_hit = [&] {
    const auto min_element = std::min_element(hits.begin(), hits.end());
    return (min_element == hits.end()) ? 0 : *min_element;
  }();
  const auto max_hit = [&] {
    const auto max_element = std::max_element(hits.begin(), hits.end());
    return (max_element == hits.end()) ? 0 : *max_element;
  }();
  const auto total_hit = std::accumulate(hits.begin(), hits.end(), uint32_t{0});
  return {min_hit, max_hit, total_hit, static_cast<float>(total_hit) / cache_size, cache_size};
}

/* ----------------------------------------------------------------------------------------------------------------- */

std::ostream& operator<<(std::ostream& o, const CacheFunctionCommon::CacheStat& cacheStat) {
  o << cacheStat.total_hit << " hits on cache with " << cacheStat.cache_size << " entries\n";
  o << "hit average: " << cacheStat.mean_hit << "in [ " << cacheStat.min_hit << " - " << cacheStat.max_hit << " ]";
  return o;
}
