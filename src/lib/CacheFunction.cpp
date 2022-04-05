#include "libKriging/CacheFunction.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

/* ----------------------------------------------------------------------------------------------------------------- */

CacheFunctionCommon::~CacheFunctionCommon() {
  //  if (!m_cache_hit.empty())
  //    std::cout << stat() << std::endl;
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
  return {min_hit, max_hit, total_hit, cache_size, m_hash_timer / 1, m_lookup_timer / 1, m_eval_timer / 1};
}

/* ----------------------------------------------------------------------------------------------------------------- */

std::ostream& operator<<(std::ostream& out, const CacheFunctionCommon::CacheStat& st) {
  std::ostream o(out.rdbuf());  // safe RAII flags restore
  o << st.total_hit << " hits on cache with " << st.cache_size << " entries\n";
  o.precision(3);
  o << "hit average: " << (1.0 * st.total_hit / st.cache_size) << " in range [ " << st.min_hit << " - " << st.max_hit
    << " ]\n";
  o.precision(1);
  o << "hash time   (ns): " << st.hash_time << " (" << (100.0 * st.hash_time / st.eval_time) << "%)\n";
  o << "lookup time (ns): " << st.lookup_time << " (" << (100.0 * st.lookup_time / st.eval_time) << "%)\n";
  o << "call time   (ns): " << st.eval_time << "\n";
  const auto current_cost = st.eval_time + st.lookup_time + st.hash_time;
  const auto nominal_cost = 1. * st.eval_time / st.cache_size * st.total_hit;
  o.precision(3);
  o << "speed up        : " << (nominal_cost / current_cost) << "\n";
  return out;
}
