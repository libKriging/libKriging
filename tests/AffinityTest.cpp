// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Optim.hpp"
// clang-format on

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

// The thread budget counts the physical cores of the CPUs the process may run
// on (docs/dev/parallel.md, section 2): restricted to one logical CPU, it is 1.
// Its own executable: the core count is read once, at the first call of the
// library, so the mask must be set before any other test runs (#473: on
// Windows the whole machine was counted, budget 10 under start /affinity 3).

namespace {
// Restricts the process to the lowest CPU it may run on; false if not possible.
bool restrict_to_one_cpu() {
#if defined(_WIN32)
  DWORD_PTR process_mask = 0, system_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) || process_mask == 0)
    return false;  // several processor groups
  const DWORD_PTR one = process_mask & (~process_mask + 1);
  return SetProcessAffinityMask(GetCurrentProcess(), one) != 0;
#elif defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return false;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &mask)) {
      cpu_set_t one;
      CPU_ZERO(&one);
      CPU_SET(cpu, &one);
      return sched_setaffinity(0, sizeof(one), &one) == 0;
    }
  return false;
#else
  return false;  // macOS: no affinity masks
#endif
}
}  // namespace

TEST_CASE("The thread budget counts only the cores of the affinity mask", "[threads]") {
  if (!restrict_to_one_cpu()) {
    WARN("affinity mask not settable here: skipped");
    return;
  }
  CHECK(Optim::get_num_threads() == 1);
}
