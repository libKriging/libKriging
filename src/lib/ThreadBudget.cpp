#include "ThreadBudget.hpp"

#include "libKriging/Optim.hpp"

#include "BlasThreads.hpp"

#include <algorithm>
#include <cmath>
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

bool has_token(const std::string& list, const std::string& token) {
  std::istringstream in(list);
  std::string t;
  while (std::getline(in, t, ','))
    if (t == token)
      return true;
  return false;
}

// CPUs allowed by the quota of one cgroup directory, 0 when unlimited or
// unreadable. v2: cpu.max "<quota> <period>" ("max <period>": unlimited);
// v1: cpu.cfs_quota_us (-1: unlimited) / cpu.cfs_period_us.
int cgroup_dir_limit(const std::string& dir) {
  {
    std::ifstream f(dir + "/cpu.max");
    std::string quota;
    long period = 0;
    if (f >> quota >> period) {
      if (quota == "max" || period <= 0)
        return 0;
      const double q = std::stod(quota);
      return std::max(1, static_cast<int>(std::ceil(q / static_cast<double>(period))));
    }
  }
  const long quota = read_long(dir + "/cpu.cfs_quota_us");
  const long period = read_long(dir + "/cpu.cfs_period_us");
  if (quota <= 0 || period <= 0)
    return 0;
  return std::max(1, static_cast<int>(std::ceil(static_cast<double>(quota) / static_cast<double>(period))));
}

// Root (within the hierarchy) and mount point of the cgroup v2 hierarchy
// (v2 = true) or of the v1 hierarchy holding the cpu controller
// (/proc/self/mountinfo); false if not mounted or unreadable.
bool cgroup_mount(bool v2, std::string& root, std::string& point) {
  std::ifstream f("/proc/self/mountinfo");
  std::string line;
  while (std::getline(f, line)) {
    const std::size_t sep = line.find(" - ");
    if (sep == std::string::npos)
      continue;
    std::istringstream pre(line.substr(0, sep)), post(line.substr(sep + 3));
    std::string id, parent, dev, r, p, fstype, source, opts;
    if (!(pre >> id >> parent >> dev >> r >> p) || !(post >> fstype >> source >> opts))
      continue;
    if (v2 ? fstype != "cgroup2" : (fstype != "cgroup" || !has_token(opts, "cpu")))
      continue;
    root = r;
    point = p;
    return true;
  }
  return false;
}

// CPUs allowed by the cgroup CPU quotas of the process, 0 when unlimited or
// unreadable: the smallest quota of its cgroup and of the ancestors visible
// under the mount point, in the v2 hierarchy and in the v1 hierarchy of the
// cpu controller (both are read: a hybrid system has both mounted).
int cgroup_cpu_limit() {
  int limit = 0;
  std::ifstream f("/proc/self/cgroup");
  std::string line;
  while (std::getline(f, line)) {  // "<id>:<controllers>:<path>"
    const std::size_t a = line.find(':');
    const std::size_t b = a == std::string::npos ? a : line.find(':', a + 1);
    if (b == std::string::npos)
      continue;
    const std::string controllers = line.substr(a + 1, b - a - 1);
    std::string rel = line.substr(b + 1);
    const bool v2 = line.compare(0, a, "0") == 0 && controllers.empty();
    if (!v2 && !has_token(controllers, "cpu"))
      continue;
    std::string root = "/", point = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/cpu";
    cgroup_mount(v2, root, point);
    if (root != "/") {
      // path relative to the mount's root; a cgroup outside it is not visible
      // (another namespace): read the mount point only
      rel = rel.compare(0, root.size(), root) == 0 ? rel.substr(root.size()) : std::string();
    }
    if (rel == "/")
      rel.clear();
    for (std::string dir = point + rel;; dir.erase(dir.rfind('/'))) {
      const int q = cgroup_dir_limit(dir);
      if (q > 0 && (limit == 0 || q < limit))
        limit = q;
      if (dir.size() <= point.size() || dir.rfind('/') == std::string::npos)
        break;
    }
  }
  return limit;
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

// MKL count of the first thread that used the library (MKL_NUM_THREADS or
// MKL's default, the physical cores), 0 without MKL.
int initial_mkl_threads() {
  static const int n = std::max(0, get_mkl_threads());
  return n;
}

// Count for a backend loaded with b0 threads (0: absent) in a call with
// `share` threads, 0 = leave it as it is.
int threads_for(int b0, int share) {
  if (b0 <= 0 || share <= 0)
    return 0;  // no such BLAS: nothing to set
  if (b0 <= share)
    return 0;  // the count it was loaded with (OPENBLAS_NUM_THREADS / MKL_NUM_THREADS) fits: left as it is
  return share;
}

thread_local int tl_share = 0;  // 0: not in a pool worker
thread_local bool tl_in_pool = false;

}  // namespace

int physical_cores() {
  static const int n = [] {
    const int os = physical_cores_os();
    return os > 0 ? std::min(os, logical_cpus()) : logical_cpus();
  }();
  return n;
}

int budget_total() {
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

int blas_threads_for(int share) {
  // Loaded with more threads than the share (its default: the logical CPUs):
  // lowering the count at run time is as fast as OPENBLAS_NUM_THREADS once
  // the threads created at load have gone to sleep (~100 ms after loading:
  // bench/thread-budget/RESULTS.md, section 9).
  return threads_for(initial_blas_threads(), share);
}

int mkl_threads_for(int share) {
  return threads_for(initial_mkl_threads(), share);
}

int omp_threads() {
  return tl_share > 0 ? tl_share : budget_total();
}

bool in_pool_worker() {
  return tl_in_pool;
}

ScopedShare::ScopedShare(int omp)
    : saved_share(tl_share), saved_in_pool(tl_in_pool), mkl(mkl_threads_for(std::max(1, omp))) {
  tl_share = std::max(1, omp);
  tl_in_pool = true;
}

ScopedShare::~ScopedShare() {
  tl_share = saved_share;
  tl_in_pool = saved_in_pool;
}

ScopedMklThreads::ScopedMklThreads(int n) {
  if (n <= 0 || get_mkl_threads() == n)
    return;
  const int previous = set_mkl_threads_local(n);
  if (previous >= 0) {
    saved = previous;
    set = true;
  }
}

ScopedMklThreads::~ScopedMklThreads() {
  if (set)
    set_mkl_threads_local(saved);
}

ScopedBlasThreads::ScopedBlasThreads(int n) {
  if (n <= 0)
    return;
  const int current = get_openblas_threads();
  if (current > 0 && current != n) {
    saved = current;
    set_openblas_threads(n);
  }
}

ScopedBlasThreads::~ScopedBlasThreads() {
  if (saved > 0)
    set_openblas_threads(saved);
}

BudgetScope::BudgetScope()
    : blas(in_pool_worker() ? 0 : blas_threads_for(budget_total())),
      mkl(in_pool_worker() ? 0 : mkl_threads_for(budget_total())) {}

}  // namespace detail
}  // namespace libKriging
