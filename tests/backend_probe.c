/* BLAS / LAPACK call probe for BlasBackendTest (ELF only, linked into the test
 * executable only).
 *
 * Defines dpotrf_ (the Cholesky factorization of every log-likelihood
 * evaluation) in the executable: on ELF the executable comes first in the
 * global lookup scope, so the library's calls to dpotrf_ land here, are
 * recorded with the calling thread and the BLAS thread counts in effect in
 * that thread (MKL_Get_Max_Threads: thread-local count if set, else global;
 * openblas_get_num_threads: process-global), then are forwarded to the next
 * definition (RTLD_NEXT: the MKL preloaded or linked, OpenBLAS, reference
 * LAPACK). Written in C so that it does not see Armadillo's own prototype.
 *
 * The integer arguments are passed through as pointers (LP64 or ILP64 alike);
 * the trailing hidden Fortran string length is forwarded as received. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>

int MKL_Get_Max_Threads(void) __attribute__((weak));
int openblas_get_num_threads(void) __attribute__((weak));

typedef void (*lk_potrf_t)(const char*, const void*, double*, const void*, void*, size_t);

enum { LK_PROBE_CAPACITY = 1 << 16 };

static pthread_mutex_t lk_probe_mutex = PTHREAD_MUTEX_INITIALIZER;
static int lk_probe_on = 0;
static int lk_probe_n = 0;
static int lk_probe_calls = 0; /* forwarded calls, recording or not */
static unsigned long lk_probe_tid[LK_PROBE_CAPACITY];
static int lk_probe_mkl[LK_PROBE_CAPACITY];
static int lk_probe_ob[LK_PROBE_CAPACITY];

static lk_potrf_t lk_next_potrf(void) {
  static lk_potrf_t next = NULL;
  if (next == NULL)
    next = (lk_potrf_t)dlsym(RTLD_NEXT, "dpotrf_");
  return next;
}

void dpotrf_(const char* uplo, const void* n, double* a, const void* lda, void* info, size_t uplo_len) {
  const int mkl = MKL_Get_Max_Threads ? MKL_Get_Max_Threads() : 0;
  const int ob = openblas_get_num_threads ? openblas_get_num_threads() : 0;
  pthread_mutex_lock(&lk_probe_mutex);
  ++lk_probe_calls;
  if (lk_probe_on && lk_probe_n < LK_PROBE_CAPACITY) {
    lk_probe_tid[lk_probe_n] = (unsigned long)pthread_self();
    lk_probe_mkl[lk_probe_n] = mkl;
    lk_probe_ob[lk_probe_n] = ob;
    ++lk_probe_n;
  }
  pthread_mutex_unlock(&lk_probe_mutex);
  lk_next_potrf()(uplo, n, a, lda, info, uplo_len);
}

/* 1 when a dpotrf_ to forward to was found */
int lk_probe_available(void) {
  return lk_next_potrf() != NULL;
}

/* Name of the shared object that serves the forwarded dpotrf_ */
const char* lk_probe_potrf_provider(void) {
  Dl_info info;
  lk_potrf_t next = lk_next_potrf();
  if (next == NULL || dladdr((void*)next, &info) == 0 || info.dli_fname == NULL)
    return "";
  return info.dli_fname;
}

void lk_probe_start(void) {
  pthread_mutex_lock(&lk_probe_mutex);
  lk_probe_n = 0;
  lk_probe_calls = 0;
  lk_probe_on = 1;
  pthread_mutex_unlock(&lk_probe_mutex);
}

void lk_probe_stop(void) {
  pthread_mutex_lock(&lk_probe_mutex);
  lk_probe_on = 0;
  pthread_mutex_unlock(&lk_probe_mutex);
}

/* Records kept since lk_probe_start (at most LK_PROBE_CAPACITY) */
int lk_probe_count(void) {
  return lk_probe_n;
}

/* Calls seen since lk_probe_start, recorded or not */
int lk_probe_total_calls(void) {
  return lk_probe_calls;
}

void lk_probe_get(int i, unsigned long* tid, int* mkl, int* openblas) {
  *tid = lk_probe_tid[i];
  *mkl = lk_probe_mkl[i];
  *openblas = lk_probe_ob[i];
}

unsigned long lk_probe_self(void) {
  return (unsigned long)pthread_self();
}
