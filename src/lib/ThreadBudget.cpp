#include "ThreadBudget.hpp"

#include "libKriging/Optim.hpp"

#include "BlasThreads.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <set>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <sched.h>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#endif

namespace libKriging {
namespace detail {

namespace {

int logical_cpus() {
  const unsigned int n = std::thread::hardware_concurrency();
  return n > 0 ? static_cast<int>(n) : 1;
}

#if defined(__linux__)
// Integer in a sysfs / procfs file, -1 if unreadable.
long read_long(const std::string& path) {
  std::ifstream f(path);
  long v = -1;
  if (!(f >> v))
    return -1;
  return v;
}

// CPUs allowed by the cgroup v2 quota (cpu.max: "<quota> <period>" or "max ..."),
// 0 when unlimited or unreadable.
int cgroup_cpu_limit() {
  std::string rel;  // "0::/some/path" -> "/some/path"
  {
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    while (std::getline(f, line))
      if (line.rfind("0::", 0) == 0)
        rel = line.substr(3);
  }
  for (const std::string& path :
       {std::string("/sys/fs/cgroup") + rel + "/cpu.max", std::string("/sys/fs/cgroup/cpu.max")}) {
    std::ifstream f(path);
    std::string quota;
    long period = 0;
    if (f >> quota >> period) {
      if (quota == "max" || period <= 0)
        return 0;
      const double q = std::stod(quota);
      return std::max(1, static_cast<int>(std::ceil(q / static_cast<double>(period))));
    }
  }
  return 0;
}

int physical_cores_os() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return 0;
  // one entry per (package, core) among the allowed CPUs
  std::set<std::pair<long, long>> cores;
  int allowed = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask))
      continue;
    ++allowed;
    const std::string topo = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const long core = read_long(topo + "core_id");
    const long pkg = read_long(topo + "physical_package_id");
    if (core < 0)
      return 0;  // no topology (some containers / VMs): fall back
    cores.insert({pkg, core});
  }
  int n = static_cast<int>(cores.size());
  if (n == 0)
    return 0;
  // a CPU quota of q CPUs is at most q cores' worth of time
  const int quota = cgroup_cpu_limit();
  if (quota > 0)
    n = std::min(n, quota);
  (void)allowed;
  return n;
}
#elif defined(__APPLE__)
int physical_cores_os() {
  int n = 0;
  size_t size = sizeof(n);
  if (sysctlbyname("hw.physicalcpu", &n, &size, nullptr, 0) != 0)
    return 0;
  return n;
}
#elif defined(_WIN32)
int physical_cores_os() {
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len == 0)
    return 0;
  std::vector<char> buf(len);
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
    return 0;
  int n = 0;
  for (DWORD off = 0; off < len;) {
    auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (p->Relationship == RelationProcessorCore)
      ++n;
    off += p->Size;
  }
  return n;
}
#else
int physical_cores_os() {
  return 0;
}
#endif

// OpenBLAS count when the library was first used (OPENBLAS_NUM_THREADS or
// OpenBLAS's default, the logical CPUs), 0 without OpenBLAS.
int initial_blas_threads() {
  static const int n = get_openblas_threads();
  return n;
}

thread_local int tl_share = 0;  // 0: not in a pool worker
thread_local bool tl_in_pool = false;
thread_local int tl_call_depth = 0;  // BudgetScopes open in this (non-pool) thread

// Process-global state of the OpenBLAS count, shared by the scopes of every
// thread (docs/dev/rationalization/T6-thread-safety.md). The count is the
// smallest one requested by the live ScopedBlasThreads; the first scope saves
// the count it finds, the last one restores it. Never destroyed (scopes may
// close during static destruction).
struct BlasCountState {
  std::mutex m;
  std::multiset<int> requests;  // counts requested by the live scopes
  int saved = 0;                // count before the first live scope, 0: no OpenBLAS
  int applied = 0;              // count currently set
  int calls = 0;                // concurrent top-level calls (BudgetScope)
  unsigned generation = 0;      // bumped in a fork child: drops the parent's scopes
};

BlasCountState& blas_state() {
  static BlasCountState* s = new BlasCountState;
  return *s;
}

// Lock-free copy of BlasCountState::calls for budget_total().
std::atomic<int> g_calls{0};

// Sets the count to the smallest live request, or back to the saved count.
void apply_locked(BlasCountState& s) {
  if (s.saved <= 0)
    return;  // no OpenBLAS
  const int target = s.requests.empty() ? s.saved : *s.requests.begin();
  if (target != s.applied) {
    set_openblas_threads(target);
    s.applied = target;
  }
}

}  // namespace

int physical_cores() {
  static const int n = [] {
    const int os = physical_cores_os();
    return os > 0 ? std::min(os, logical_cpus()) : logical_cpus();
  }();
  return n;
}

int budget_full() {
  double share = Optim::num_threads_percent;
  if (!(share > 0.0))
    share = 1.0;
  else if (share > 1.0)
    share /= 100.0;
  int total = static_cast<int>(std::lround(share * physical_cores()));
  total = std::min(total, logical_cpus());
#ifdef _OPENMP
  const int omp_limit = omp_get_max_threads();
  if (omp_limit > 0 && !tl_in_pool)  // pool workers: their share is set below
    total = std::min(total, omp_limit);
#endif
  // The OpenBLAS count (OPENBLAS_NUM_THREADS) bounds the BLAS only, not the
  // pool and the OpenMP loops: see blas_threads_for().
  return std::max(1, total);
}

int budget_total() {
  const int full = budget_full();
  const int calls = tl_in_pool ? 1 : g_calls.load(std::memory_order_relaxed);
  return calls > 1 ? std::max(1, full / calls) : full;
}

int concurrent_calls() {
  return g_calls.load(std::memory_order_relaxed);
}

int blas_threads_for(int share) {
  const int b0 = initial_blas_threads();
  if (b0 <= 0 || share <= 0)
    return 0;  // no OpenBLAS: nothing to set
  if (b0 <= share)
    return 0;  // the count it was loaded with (OPENBLAS_NUM_THREADS) fits: left as it is
  // Loaded with more threads than the share (its default: the logical CPUs).
  // Lowering the count at run time is as fast as OPENBLAS_NUM_THREADS once
  // the threads created at load have gone to sleep (~100 ms after loading:
  // bench/thread-budget/RESULTS.md, section 9).
  return share;
}

int omp_threads() {
  return tl_share > 0 ? tl_share : budget_total();
}

bool in_pool_worker() {
  return tl_in_pool;
}

ScopedShare::ScopedShare(int omp) : saved_share(tl_share), saved_in_pool(tl_in_pool) {
  tl_share = std::max(1, omp);
  tl_in_pool = true;
}

ScopedShare::~ScopedShare() {
  tl_share = saved_share;
  tl_in_pool = saved_in_pool;
}

ScopedBlasThreads::ScopedBlasThreads(int n) {
  if (n <= 0)
    return;
  BlasCountState& s = blas_state();
  std::lock_guard<std::mutex> lock(s.m);
  if (s.requests.empty()) {
    s.saved = get_openblas_threads();
    s.applied = s.saved;
  }
  s.requests.insert(n);
  requested = n;
  generation = s.generation;
  apply_locked(s);
}

ScopedBlasThreads::~ScopedBlasThreads() {
  if (requested <= 0)
    return;
  BlasCountState& s = blas_state();
  std::lock_guard<std::mutex> lock(s.m);
  if (generation != s.generation)
    return;  // opened before a fork, in the parent
  const auto it = s.requests.find(requested);
  if (it != s.requests.end())
    s.requests.erase(it);
  apply_locked(s);
  if (s.requests.empty())
    s.saved = s.applied = 0;
}

CallCount::CallCount() : counted(!tl_in_pool) {
  if (!counted)
    return;
  if (tl_call_depth++ > 0)
    return;  // nested call of the same thread: counted once
  BlasCountState& s = blas_state();
  std::lock_guard<std::mutex> lock(s.m);
  ++s.calls;
  g_calls.store(s.calls, std::memory_order_relaxed);
  generation = s.generation;
  outermost = true;
}

CallCount::~CallCount() {
  if (!counted)
    return;
  --tl_call_depth;
  if (!outermost)
    return;
  BlasCountState& s = blas_state();
  std::lock_guard<std::mutex> lock(s.m);
  if (generation != s.generation)
    return;
  --s.calls;
  g_calls.store(s.calls, std::memory_order_relaxed);
}

BudgetScope::BudgetScope() : blas(in_pool_worker() ? 0 : blas_threads_for(budget_total())) {}

void blas_count_fork_prepare() {
  blas_state().m.lock();
}

void blas_count_fork_parent() {
  blas_state().m.unlock();
}

void blas_count_fork_child() {
  // Only the forking thread survives: the scopes of the others never close.
  BlasCountState& s = blas_state();
  s.requests.clear();
  s.saved = s.applied = 0;
  s.calls = 0;
  g_calls.store(0, std::memory_order_relaxed);
  ++s.generation;
  s.m.unlock();
}

}  // namespace detail
}  // namespace libKriging
