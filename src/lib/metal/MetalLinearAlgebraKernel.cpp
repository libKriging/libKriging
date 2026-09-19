// UNVERIFIED Apple-Metal port of src/lib/cuda/CudaLinearAlgebraKernel.cu.
// Uses metal-cpp (headers only, no Objective-C); the MSL kernels
// (lk_metal_kernels.metal, embedded via the generated lk_metal_kernels_msl.hpp)
// are compiled at runtime. FLOAT32 throughout -- see MetalLinearAlgebra.hpp.
// Never compiled or run: no macOS / Metal toolchain in the environment this
// was written in.
//
// Confines every Metal / metal-cpp symbol to this TU; the plain-C lk_metal_*
// surface (MetalLinearAlgebraKernel.hpp) is all MetalLinearAlgebra.cpp sees,
// same ABI-safety rule as the CUDA backend. Device "pointers" handed across
// that surface are actually MTL::Buffer* handles cast to void*; the host
// orchestration only ever passes them through, never dereferences them.
//
// Each launch commits its own command buffer and waits -- simple and
// correct; a verified follow-up should batch a CG iteration's kernels into
// one command buffer and only wait at the sync points. double values at the
// boundary are narrowed to float on upload and widened back on download.

#include "MetalLinearAlgebraKernel.hpp"

#ifdef LIBKRIGING_USE_METAL_ITERATIVE

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "lk_metal_kernels_msl.hpp"  // generated: lk_metal_kernels_msl()

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Ctx {
  MTL::Device* dev = nullptr;
  MTL::CommandQueue* queue = nullptr;
  MTL::Library* lib = nullptr;
  std::map<std::string, MTL::ComputePipelineState*> pso;

  Ctx() {
    dev = MTL::CreateSystemDefaultDevice();
    if (!dev)
      throw std::runtime_error("lk_metal: no Metal device");
    queue = dev->newCommandQueue();
    NS::Error* err = nullptr;
    auto* src = NS::String::string(lk_metal_kernels_msl(), NS::UTF8StringEncoding);
    lib = dev->newLibrary(src, nullptr, &err);
    if (!lib)
      throw std::runtime_error("lk_metal: MSL compile failed");
  }

  MTL::ComputePipelineState* get(const char* name) {
    auto it = pso.find(name);
    if (it != pso.end())
      return it->second;
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn)
      throw std::runtime_error(std::string("lk_metal: no kernel '") + name + "'");
    NS::Error* err = nullptr;
    auto* p = dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!p)
      throw std::runtime_error(std::string("lk_metal: pipeline for '") + name + "' failed");
    pso[name] = p;
    return p;
  }
};

Ctx& ctx() {
  static Ctx c;  // constructed on first use; throws if no device
  return c;
}

const bool g_available = [] {
  try {
    (void)ctx();
    return true;
  } catch (...) {
    return false;
  }
}();

inline MTL::Buffer* buf(void* p) {
  return static_cast<MTL::Buffer*>(p);
}
inline MTL::Buffer* buf(const void* p) {
  return static_cast<MTL::Buffer*>(const_cast<void*>(p));
}

// One 1D or 2D dispatch of `name`, binding the device buffers in `bufs`
// (index 0..) then `bytes` as a constant at index `bufs.size()`.
void run(const char* name, const std::vector<const void*>& bufs, const void* params, std::size_t params_size, int gx,
         int gy) {
  auto* pso = ctx().get(name);
  auto* cb = ctx().queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(pso);
  for (std::size_t i = 0; i < bufs.size(); ++i)
    enc->setBuffer(buf(bufs[i]), 0, static_cast<NS::UInteger>(i));
  if (params)
    enc->setBytes(params, params_size, static_cast<NS::UInteger>(bufs.size()));
  const NS::UInteger maxw = pso->maxTotalThreadsPerThreadgroup();
  NS::UInteger tgx = (gy > 1) ? 32 : (maxw < 64 ? maxw : 64);
  NS::UInteger tgy = (gy > 1) ? (maxw / tgx ? maxw / tgx : 1) : 1;
  enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(gx), static_cast<NS::UInteger>(gy), 1),
                       MTL::Size(tgx, tgy, 1));
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
}

struct RmulP {
  int n, dimX, covKind, ncols;
};
struct NC {
  int n, ncols;
};
struct CgP {
  float tol;
  int ncols;
};
struct PcP {
  int n, k, ncols;
};

}  // namespace

extern "C" int lk_metal_available(void) {
  return g_available ? 1 : 0;
}

extern "C" void* lk_metal_malloc(unsigned long bytes) {
  auto* b = ctx().dev->newBuffer(static_cast<NS::UInteger>(bytes ? bytes : 1), MTL::ResourceStorageModeShared);
  if (!b)
    throw std::runtime_error("lk_metal_malloc failed");
  return b;
}
extern "C" void lk_metal_free(void* p) {
  if (p)
    buf(p)->release();
}
extern "C" void lk_metal_upload_f64_as_f32(void* dbuf, const double* host, unsigned long count) {
  float* dst = static_cast<float*>(buf(dbuf)->contents());
  for (unsigned long i = 0; i < count; ++i)
    dst[i] = static_cast<float>(host[i]);
}
extern "C" void lk_metal_download_f32_as_f64(double* host, const void* dbuf, unsigned long count) {
  const float* src = static_cast<const float*>(buf(dbuf)->contents());
  for (unsigned long i = 0; i < count; ++i)
    host[i] = static_cast<double>(src[i]);
}
extern "C" void lk_metal_upload_i32(void* dbuf, const int* host, unsigned long count) {
  std::memcpy(buf(dbuf)->contents(), host, count * sizeof(int));
}
extern "C" void lk_metal_download_i32(int* host, const void* dbuf, unsigned long count) {
  std::memcpy(host, buf(dbuf)->contents(), count * sizeof(int));
}
extern "C" void lk_metal_memset_dev(void* dbuf, int value, unsigned long bytes) {
  std::memset(buf(dbuf)->contents(), value, static_cast<std::size_t>(bytes));
}
extern "C" void lk_metal_copy_dev(void* dst, const void* src, unsigned long bytes) {
  std::memcpy(buf(dst)->contents(), buf(src)->contents(), static_cast<std::size_t>(bytes));
}

extern "C" int lk_metal_rmul_batched_scratch_elems(int, int) {
  return 0;
}

extern "C" void lk_metal_rmul_batched_launch(const void* Xt, int n, int dimX, const void* theta, int covKind,
                                             const void* P, int ncols, void* Ap, void*) {
  RmulP p{n, dimX, covKind, ncols};
  run("rmul_batched", {Xt, theta, P, Ap}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_drmul_batched_launch(const void* Xt, int n, int dimX, const void* theta, int covKind,
                                              const void* V, int ncols, void* Out) {
  RmulP p{n, dimX, covKind, ncols};
  run("drmul_batched", {Xt, theta, V, Out}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_batched_dot_launch(const void* A, const void* B, int n, int ncols, void* out) {
  NC p{n, ncols};
  run("batched_dot", {A, B, out}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_batched_axpy_launch(const void* alpha, const void* X, void* Y, int n, int ncols) {
  NC p{n, ncols};
  run("batched_axpy", {alpha, X, Y}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_batched_update_p_launch(const void* R, const void* beta, void* P, int n, int ncols) {
  NC p{n, ncols};
  run("batched_update_p", {R, beta, P}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_cg_alpha_launch(const void* rz_old, const void* pAp, int ncols, void* active, void* alpha,
                                         void* neg_alpha) {
  CgP p{0.0f, ncols};
  run("cg_alpha", {rz_old, pAp, active, alpha, neg_alpha}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_beta_launch(const void* rr_new, const void* bnorm, double tol, int ncols, void* active,
                                        void* rz_old, void* beta) {
  CgP p{static_cast<float>(tol), ncols};
  run("cg_beta", {rr_new, bnorm, active, rz_old, beta}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_restart_launch(const void* rr, const void* bnorm, double tol, int ncols, void* active,
                                           void* rz_old) {
  CgP p{static_cast<float>(tol), ncols};
  run("cg_restart", {rr, bnorm, active, rz_old}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_restart_precond_launch(const void* rr, const void* rz, const void* bnorm, double tol,
                                                   int ncols, void* active, void* rz_old) {
  CgP p{static_cast<float>(tol), ncols};
  run("cg_restart_precond", {rr, rz, bnorm, active, rz_old}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_beta_precond_launch(const void* rr, const void* rz_new, const void* bnorm, double tol,
                                                int ncols, void* active, void* rz_old, void* beta) {
  CgP p{static_cast<float>(tol), ncols};
  run("cg_beta_precond", {rr, rz_new, bnorm, active, rz_old, beta}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_any_active_launch(const void* active, int ncols, void* flag) {
  run("cg_any_active", {active, flag}, &ncols, sizeof(ncols), ncols, 1);
}
extern "C" void lk_metal_precond_apply_launch(const void* U, int n, int k, const void* Dinv, const void* Mchol,
                                              const void* r, int ncols, void* z, void* scratch_nc, void* scratch_kc) {
  PcP p{n, k, ncols};
  run("precond_scale_rows", {Dinv, r, scratch_nc}, &p, sizeof(p), n, ncols);
  run("precond_gemm_Ut", {U, scratch_nc, scratch_kc}, &p, sizeof(p), k, ncols);
  run("precond_trisolve", {Mchol, scratch_kc}, &p, sizeof(p), ncols, 1);
  run("precond_combine", {U, Dinv, r, scratch_kc, z}, &p, sizeof(p), n, ncols);
}

#endif  // LIBKRIGING_USE_METAL_ITERATIVE
