// Standalone reproducer (no libKriging): Accelerate dpotri on a gauss correlation matrix.
// usage: potri_repro n calls [theta1 theta2 theta3] [--stress k] [--single]
//   builds R(i,j) = exp(-0.5 sum_k ((x_ik - x_jk)/theta_k)^2) + 1e-9 I on the Weyl design of
//   tests/BlasBackendTest.cpp, L = dpotrf(R); reference inverse = dpotrs(L, I) (two triangular solves);
//   calls dpotri on a fresh copy of L `calls` times and counts results farther than 1e-8 (relative to
//   max|ref|) from the reference.  --stress k: k threads doing dgemm in the background;
//   --single: BLASSetThreading(single) (macOS 15+).
#include <Accelerate/Accelerate.h>
#include <vecLib/thread_api.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
extern "C" {
void dpotrf_(const char*, const int*, double*, const int*, int*);
void dpotri_(const char*, const int*, double*, const int*, int*);
void dpotrs_(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*);
}
int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 750, calls = argc > 2 ? atoi(argv[2]) : 1000;
  double th[3] = {0.3, 0.5, 0.7};
  int stress = 0; bool single = false;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--stress") && i + 1 < argc) stress = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--single")) single = true;
    else if (i + 2 < argc + 0 && argv[i][0] != '-' && i + 2 < argc) { th[0] = atof(argv[i]); th[1] = atof(argv[i+1]); th[2] = atof(argv[i+2]); i += 2; }
  }
  if (single) { if (__builtin_available(macOS 15.0, *)) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED); }
  const double a[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898};
  std::vector<double> X(3 * n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < 3; ++j) X[j * n + i] = std::fmod(0.5 + (i + 1) * a[j], 1.0);
  std::vector<double> R((size_t)n * n), L;
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
    double s = 0; for (int k = 0; k < 3; ++k) { double d = (X[k*n+i] - X[k*n+j]) / th[k]; s += d * d; }
    R[(size_t)j * n + i] = std::exp(-0.5 * s) + (i == j ? 1e-9 : 0.0);
  }
  L = R; int info = 0; const char lo = 'L';
  dpotrf_(&lo, &n, L.data(), &n, &info);
  if (info) { printf("dpotrf info=%d\n", info); return 1; }
  std::vector<double> ref((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) ref[(size_t)i * n + i] = 1.0;
  dpotrs_(&lo, &n, &n, L.data(), &n, ref.data(), &n, &info);
  double scale = 0; for (double v : ref) scale = std::fmax(scale, std::fabs(v));
  std::atomic<bool> stop{false};
  std::vector<std::thread> bg;
  for (int t = 0; t < stress; ++t) bg.emplace_back([&] {
    const int m = 400; std::vector<double> A(m*m, 1.0), B(m*m, 1.0), C(m*m, 0.0);
    while (!stop.load()) cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, A.data(), m, B.data(), m, 0.0, C.data(), m);
  });
  int faults = 0, infos = 0; double worst = 0, worst_fault = 0; std::vector<double> P;
  auto t0 = std::chrono::steady_clock::now();
  for (int c = 0; c < calls; ++c) {
    P = L; info = 0;
    dpotri_(&lo, &n, P.data(), &n, &info);
    if (info) { ++infos; ++faults; continue; }
    double gap = 0; // lower triangle vs ref
    for (int j = 0; j < n; ++j) for (int i = j; i < n; ++i) gap = std::fmax(gap, std::fabs(P[(size_t)j*n+i] - ref[(size_t)j*n+i]));
    gap /= scale; if (!(gap <= 1e-8)) { ++faults; worst_fault = std::fmax(worst_fault, gap); }
    worst = std::fmax(worst, gap);
  }
  stop.store(true); for (auto& t : bg) t.join();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("n=%d theta=(%g,%g,%g) calls=%d stress=%d single=%d: %d faults (info!=0: %d), worst gap %.3g, worst fault %.3g, %.1f ms/call\n",
         n, th[0], th[1], th[2], calls, stress, (int)single, faults, infos, worst, worst_fault, 1e3 * sec / calls);
  return faults ? 2 : 0;
}
