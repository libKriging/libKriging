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
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Command-buffer batching: a CG iteration issues a dozen-odd tiny kernels
// (matvec, dot, alpha/beta scalar updates, axpy, precond apply's 4
// sub-kernels...) that are almost all GPU-GPU dependent -- only the host's
// periodic "any column still active?" poll (every sync_every/restart_every
// iterations, see MetalLinearAlgebra.cpp) actually needs the result back on
// the CPU. Originally every single kernel launch committed its own command
// buffer and called waitUntilCompleted() -- correct, but it serialized the
// whole CG loop on a full host<->GPU round-trip PER KERNEL (measured: ~40s
// for a single n=4000 SLQ logLik call, ~35x slower than the CPU dense path,
// vs. the CUDA/HIP backends which only sync at the same periodic points
// this file now mirrors). Fix: `run()` (compute) and copy_dev/memset_dev
// (blit) all encode onto a single pending command buffer/encoder instead of
// committing immediately; only upload_*/download_* -- the only calls that
// ever hand a value back to the CPU or take one from it -- flush (end any
// open encoder, commit, waitUntilCompleted) first. Encode-order dependencies
// between GPU commands are still resolved correctly: Metal's default hazard
// tracking serializes same-command-buffer commands that read what an
// earlier one in the buffer wrote, exactly like CUDA's in-order stream.
struct Ctx {
  MTL::Device* dev = nullptr;
  MTL::CommandQueue* queue = nullptr;
  MTL::Library* lib = nullptr;
  std::map<std::string, MTL::ComputePipelineState*> pso;
  std::mutex mtx;
  MTL::CommandBuffer* pend_cb = nullptr;
  MTL::ComputeCommandEncoder* pend_compute = nullptr;
  MTL::BlitCommandEncoder* pend_blit = nullptr;

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

  // Caller holds mtx. Ends whichever encoder kind is open if it doesn't
  // match, (re)opens a command buffer if needed, and returns the compute
  // encoder to encode onto.
  MTL::ComputeCommandEncoder* ensure_compute() {
    if (pend_blit) {
      pend_blit->endEncoding();
      pend_blit = nullptr;
    }
    if (!pend_cb)
      pend_cb = queue->commandBuffer();
    if (!pend_compute)
      pend_compute = pend_cb->computeCommandEncoder();
    return pend_compute;
  }

  // Caller holds mtx. Same as ensure_compute() for the blit encoder (used
  // by copy_dev/memset_dev, so a device-to-device copy or fill is ordered
  // correctly against surrounding compute dispatches within the batch).
  MTL::BlitCommandEncoder* ensure_blit() {
    if (pend_compute) {
      pend_compute->endEncoding();
      pend_compute = nullptr;
    }
    if (!pend_cb)
      pend_cb = queue->commandBuffer();
    if (!pend_blit)
      pend_blit = pend_cb->blitCommandEncoder();
    return pend_blit;
  }

  // Caller holds mtx. Ends any open encoder and commits+waits the pending
  // command buffer, if any. Called from upload_*/download_* only -- the
  // only points where a value crosses the GPU<->CPU boundary, so it's the
  // only place a flush is ever required for correctness.
  void flush() {
    if (pend_compute) {
      pend_compute->endEncoding();
      pend_compute = nullptr;
    }
    if (pend_blit) {
      pend_blit->endEncoding();
      pend_blit = nullptr;
    }
    if (pend_cb) {
      pend_cb->commit();
      pend_cb->waitUntilCompleted();
      pend_cb = nullptr;
    }
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

// A kernel buffer argument: a device buffer handle plus an optional byte
// offset into it (Metal's normal way to bind a sub-region of a buffer --
// e.g. one step's n*npr block of the Lanczos history buffer -- without a
// separate buffer object). The converting constructor lets existing
// `run(name, {a, b, c}, ...)` call sites keep compiling unchanged (each
// bare pointer becomes a BufArg with offset 0).
struct BufArg {
  const void* p;
  std::size_t byte_offset;
  BufArg(const void* ptr, std::size_t offset = 0) : p(ptr), byte_offset(offset) {}
};

// One 1D or 2D dispatch of `name`, binding the device buffers in `bufs`
// (index 0..) then `bytes` as a constant at index `bufs.size()`. Encodes
// onto the shared pending command buffer (see Ctx's batching comment) --
// does NOT commit or wait; that happens lazily in upload_*/download_*.
void run(const char* name, const std::vector<BufArg>& bufs, const void* params, std::size_t params_size, int gx,
         int gy) {
  auto* pso = ctx().get(name);
  std::lock_guard<std::mutex> lock(ctx().mtx);
  auto* enc = ctx().ensure_compute();
  enc->setComputePipelineState(pso);
  for (std::size_t i = 0; i < bufs.size(); ++i)
    enc->setBuffer(buf(bufs[i].p), static_cast<NS::UInteger>(bufs[i].byte_offset), static_cast<NS::UInteger>(i));
  if (params)
    enc->setBytes(params, params_size, static_cast<NS::UInteger>(bufs.size()));
  const NS::UInteger maxw = pso->maxTotalThreadsPerThreadgroup();
  NS::UInteger tgx = (gy > 1) ? 32 : (maxw < 64 ? maxw : 64);
  NS::UInteger tgy = (gy > 1) ? (maxw / tgx ? maxw / tgx : 1) : 1;
  enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(gx), static_cast<NS::UInteger>(gy), 1),
                       MTL::Size(tgx, tgy, 1));
}

struct RmulP {
  int n, dimX, covKind, ncols;
};
struct NC {
  int n, ncols;
};
struct CgP {
  int ncols;
};
struct PcP {
  int n, k, ncols;
};
struct LzAlphaP {
  int ncols, step_idx;
};
struct LzBetaP {
  int ncols, step_idx, is_last;
};
struct LzReP {
  int n, npr, ls, m;
};
struct BuildCovP {
  int n, dimX, covKind;
};
struct DenseMvP {
  int n, ncols, ldc;
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
// Every upload_*/download_* flushes the pending batch first (see Ctx's
// comment): uploads so a freshly-written host value isn't raced by an
// earlier-encoded-but-not-yet-run kernel that reads the same buffer, and
// downloads so the CPU-visible bytes it reads back are actually final.
extern "C" void lk_metal_upload_f64_as_f32(void* dbuf, const double* host, unsigned long count) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  ctx().flush();
  float* dst = static_cast<float*>(buf(dbuf)->contents());
  for (unsigned long i = 0; i < count; ++i)
    dst[i] = static_cast<float>(host[i]);
}
extern "C" void lk_metal_download_f32_as_f64(double* host, const void* dbuf, unsigned long count) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  ctx().flush();
  const float* src = static_cast<const float*>(buf(dbuf)->contents());
  for (unsigned long i = 0; i < count; ++i)
    host[i] = static_cast<double>(src[i]);
}
extern "C" void lk_metal_upload_i32(void* dbuf, const int* host, unsigned long count) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  ctx().flush();
  std::memcpy(buf(dbuf)->contents(), host, count * sizeof(int));
}
extern "C" void lk_metal_download_i32(int* host, const void* dbuf, unsigned long count) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  ctx().flush();
  std::memcpy(host, buf(dbuf)->contents(), count * sizeof(int));
}
// memset_dev/copy_dev encode a blit (fillBuffer/copyFromBuffer) onto the
// pending batch instead of memcpy'ing the shared buffer directly from the
// CPU -- a same-batch kernel that's supposed to read the result of an
// earlier-encoded-but-not-yet-run kernel would otherwise see stale data (a
// CPU memcpy done between two encode calls runs immediately, before either
// kernel has actually executed on the GPU).
extern "C" void lk_metal_memset_dev(void* dbuf, int value, unsigned long bytes) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  auto* enc = ctx().ensure_blit();
  enc->fillBuffer(buf(dbuf), NS::Range(0, bytes), static_cast<uint8_t>(value));
}
extern "C" void lk_metal_copy_dev(void* dst, const void* src, unsigned long bytes) {
  std::lock_guard<std::mutex> lock(ctx().mtx);
  auto* enc = ctx().ensure_blit();
  enc->copyFromBuffer(buf(src), 0, buf(dst), 0, static_cast<NS::UInteger>(bytes));
}

extern "C" int lk_metal_rmul_batched_scratch_elems(int, int) {
  return 0;
}

extern "C" void lk_metal_rmul_batched_launch(const void* Xt, int n, int dimX, const void* theta, int covKind,
                                             const void* P, int ncols, void* Ap, void*, long p_byte_offset) {
  RmulP p{n, dimX, covKind, ncols};
  run("rmul_batched", {Xt, theta, BufArg(P, static_cast<std::size_t>(p_byte_offset)), Ap}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_drmul_batched_launch(const void* Xt, int n, int dimX, const void* theta, int covKind,
                                              const void* V, int ncols, void* Out) {
  RmulP p{n, dimX, covKind, ncols};
  run("drmul_batched", {Xt, theta, V, Out}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_batched_dot_launch(const void* A, const void* B, int n, int ncols, void* out,
                                            long a_byte_offset, long b_byte_offset) {
  NC p{n, ncols};
  run("batched_dot",
     {BufArg(A, static_cast<std::size_t>(a_byte_offset)), BufArg(B, static_cast<std::size_t>(b_byte_offset)), out},
     &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_batched_axpy_launch(const void* alpha, const void* X, void* Y, int n, int ncols,
                                             long x_byte_offset, long y_byte_offset) {
  NC p{n, ncols};
  run("batched_axpy",
     {alpha, BufArg(X, static_cast<std::size_t>(x_byte_offset)), BufArg(Y, static_cast<std::size_t>(y_byte_offset))},
     &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_batched_update_p_launch(const void* R, const void* beta, void* P, int n, int ncols) {
  NC p{n, ncols};
  run("batched_update_p", {R, beta, P}, &p, sizeof(p), n, ncols);
}
extern "C" void lk_metal_cg_alpha_launch(const void* rz_old, const void* pAp, int ncols, void* active, void* alpha,
                                         void* neg_alpha) {
  CgP p{ncols};
  run("cg_alpha", {rz_old, pAp, active, alpha, neg_alpha}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_beta_launch(const void* rr_new, const void* bnorm, const void* tol, int ncols,
                                        void* active, void* rz_old, void* beta) {
  CgP p{ncols};
  run("cg_beta", {rr_new, bnorm, tol, active, rz_old, beta}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_restart_launch(const void* rr, const void* bnorm, const void* tol, int ncols,
                                           void* active, void* rz_old) {
  CgP p{ncols};
  run("cg_restart", {rr, bnorm, tol, active, rz_old}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_restart_precond_launch(const void* rr, const void* rz, const void* bnorm,
                                                   const void* tol, int ncols, void* active, void* rz_old) {
  CgP p{ncols};
  run("cg_restart_precond", {rr, rz, bnorm, tol, active, rz_old}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_cg_beta_precond_launch(const void* rr, const void* rz_new, const void* bnorm,
                                                const void* tol, int ncols, void* active, void* rz_old, void* beta) {
  CgP p{ncols};
  run("cg_beta_precond", {rr, rz_new, bnorm, tol, active, rz_old, beta}, &p, sizeof(p), ncols, 1);
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

extern "C" void lk_metal_lanczos_alpha_launch(const void* dot, int ncols, int step_idx, const void* active,
                                              void* alpha_all, void* neg_alpha) {
  LzAlphaP p{ncols, step_idx};
  run("lanczos_alpha", {dot, active, alpha_all, neg_alpha}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_lanczos_beta_launch(const void* dot2, int ncols, int step_idx, int is_last, void* active,
                                             void* m_eff, void* beta_out, void* inv_bj_out, void* neg_beta_prev_out) {
  LzBetaP p{ncols, step_idx, is_last};
  run("lanczos_beta", {dot2, active, m_eff, beta_out, inv_bj_out, neg_beta_prev_out}, &p, sizeof(p), ncols, 1);
}
extern "C" void lk_metal_lanczos_reorth_dot_launch(const void* V, const void* W, int n, int npr, int ls, int m,
                                                   void* t) {
  LzReP p{n, npr, ls, m};
  run("lanczos_reorth_dot", {V, W, t}, &p, sizeof(p), m, npr);
}
extern "C" void lk_metal_lanczos_reorth_sub_launch(const void* V, const void* t, int n, int npr, int ls, int m,
                                                   void* W) {
  LzReP p{n, npr, ls, m};
  run("lanczos_reorth_sub", {V, t, W}, &p, sizeof(p), n, npr);
}

extern "C" void lk_metal_build_cov_launch(const void* Xt, int n, int dimX, const void* theta, int covKind, void* R) {
  BuildCovP p{n, dimX, covKind};
  run("build_cov", {Xt, theta, R}, &p, sizeof(p), n, n);
}
extern "C" void lk_metal_dense_matvec_launch(const void* R, int n, const void* V, int ncols, void* Out, int ldc,
                                             long v_byte_offset) {
  DenseMvP p{n, ncols, ldc};
  run("dense_matvec", {R, BufArg(V, static_cast<std::size_t>(v_byte_offset)), Out}, &p, sizeof(p), n, ncols);
}

#endif  // LIBKRIGING_USE_METAL_ITERATIVE
