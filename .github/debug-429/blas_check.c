/* Temporary diagnostic for #429: BLAS / LAPACK results of the system library
 * (libblas.so.3 / liblapack.so.3, i.e. OpenBLAS once octave is installed)
 * against a naive long double reference.
 *
 * One line per failing case, then one summary line:
 *   SUMMARY core=<openblas core> threads=<n> cases=<n> fails=<n> worst=<case> ratio=<r>
 * ratio = observed error / (size-dependent rounding bound); > TOL is a failure.
 * Exit code 0 always (the caller compares configurations). */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int blasint;
extern void dgemm_(const char*, const char*, const blasint*, const blasint*, const blasint*, const double*,
                   const double*, const blasint*, const double*, const blasint*, const double*, double*,
                   const blasint*);
extern void dgemv_(const char*, const blasint*, const blasint*, const double*, const double*, const blasint*,
                   const double*, const blasint*, const double*, double*, const blasint*);
extern void dsyrk_(const char*, const char*, const blasint*, const blasint*, const double*, const double*,
                   const blasint*, const double*, double*, const blasint*);
extern void dtrsm_(const char*, const char*, const char*, const char*, const blasint*, const blasint*,
                   const double*, const double*, const blasint*, double*, const blasint*);
extern void dsymv_(const char*, const blasint*, const double*, const double*, const blasint*, const double*,
                   const blasint*, const double*, double*, const blasint*);
extern double ddot_(const blasint*, const double*, const blasint*, const double*, const blasint*);
extern void dpotrf_(const char*, const blasint*, double*, const blasint*, blasint*);
extern void dgesv_(const blasint*, const blasint*, double*, const blasint*, blasint*, double*, const blasint*,
                   blasint*);

#define TOL 50.0
#define EPS 2.220446049250313e-16

static int n_cases = 0, n_fails = 0;
static double worst_ratio = 0;
static char worst_case[128] = "none";

static double urand(void) { return 2.0 * rand() / (double)RAND_MAX - 1.0; }
static double* rmat(int m, int n) {
  double* a = malloc(sizeof(double) * (size_t)(m > 0 ? m : 1) * (n > 0 ? n : 1));
  for (long i = 0; i < (long)m * n; i++)
    a[i] = urand();
  return a;
}
static void record(const char* name, double ratio) {
  n_cases++;
  if (!(ratio <= TOL)) { /* also catches NaN */
    n_fails++;
    printf("FAIL %s ratio=%.3g\n", name, ratio);
  }
  if (!(ratio <= worst_ratio)) {
    worst_ratio = ratio;
    snprintf(worst_case, sizeof worst_case, "%s", name);
  }
}

/* op(X)(i,j) for a column-major X with leading dimension ld */
static double op(const double* x, int ld, char t, int i, int j) { return t == 'N' ? x[i + (long)j * ld] : x[j + (long)i * ld]; }

static void check_gemm(int m, int n, int k, char ta, char tb, double beta) {
  int lda = ta == 'N' ? m : k, ldb = tb == 'N' ? k : n;
  double* a = rmat(lda, ta == 'N' ? k : m);
  double* b = rmat(ldb, tb == 'N' ? n : k);
  double* c = rmat(m, n);
  double* c0 = malloc(sizeof(double) * (size_t)(m * n > 0 ? m * n : 1));
  memcpy(c0, c, sizeof(double) * (size_t)m * n);
  double alpha = 0.75;
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &m);
  double ratio = 0;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      long double s = 0, sa = 0;
      for (int l = 0; l < k; l++) {
        long double p = (long double)op(a, lda, ta, i, l) * op(b, ldb, tb, l, j);
        s += p;
        sa += fabsl(p);
      }
      long double ref = alpha * s + beta * c0[i + (long)j * m];
      long double bound = (fabsl(alpha) * sa + fabs(beta * c0[i + (long)j * m])) * (k + 2) * EPS + 1e-300;
      double r = (double)(fabsl(c[i + (long)j * m] - ref) / bound);
      if (!(r <= ratio)) ratio = r;
    }
  char name[128];
  snprintf(name, sizeof name, "dgemm_%c%c_m%d_n%d_k%d_beta%g", ta, tb, m, n, k, beta);
  record(name, ratio);
  free(a), free(b), free(c), free(c0);
}

static void check_gemv(int m, int n, char t) {
  double* a = rmat(m, n);
  int lx = t == 'N' ? n : m, ly = t == 'N' ? m : n, one = 1;
  double *x = rmat(lx, 1), *y = rmat(ly, 1), *y0 = malloc(sizeof(double) * ly);
  memcpy(y0, y, sizeof(double) * ly);
  double alpha = 1.25, beta = 0.5, ratio = 0;
  dgemv_(&t, &m, &n, &alpha, a, &m, x, &one, &beta, y, &one);
  for (int i = 0; i < ly; i++) {
    long double s = 0, sa = 0;
    for (int l = 0; l < lx; l++) {
      long double p = (long double)(t == 'N' ? a[i + (long)l * m] : a[l + (long)i * m]) * x[l];
      s += p, sa += fabsl(p);
    }
    long double ref = alpha * s + beta * y0[i];
    double r = (double)(fabsl(y[i] - ref) / ((alpha * sa + fabs(beta * y0[i])) * (lx + 2) * EPS + 1e-300));
    if (!(r <= ratio)) ratio = r;
  }
  char name[128];
  snprintf(name, sizeof name, "dgemv_%c_m%d_n%d", t, m, n);
  record(name, ratio);
  free(a), free(x), free(y), free(y0);
}

static double* spd(int n) { /* A = G G^T / n + I */
  double* g = rmat(n, n);
  double* a = malloc(sizeof(double) * (size_t)n * n);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++) {
      long double s = 0;
      for (int l = 0; l < n; l++)
        s += (long double)g[i + (long)l * n] * g[j + (long)l * n];
      a[i + (long)j * n] = (double)(s / n) + (i == j);
    }
  free(g);
  return a;
}

static void check_syrk(int n, int k, char uplo, char t) {
  int lda = t == 'N' ? n : k;
  double* a = rmat(lda, t == 'N' ? k : n);
  double *c = rmat(n, n), *c0 = malloc(sizeof(double) * (size_t)n * n);
  memcpy(c0, c, sizeof(double) * (size_t)n * n);
  double alpha = 1.0, beta = 0.5, ratio = 0;
  dsyrk_(&uplo, &t, &n, &k, &alpha, a, &lda, &beta, c, &n);
  for (int j = 0; j < n; j++)
    for (int i = (uplo == 'L' ? j : 0); i < (uplo == 'L' ? n : j + 1); i++) {
      long double s = 0, sa = 0;
      for (int l = 0; l < k; l++) {
        long double p = (long double)op(a, lda, t, i, l) * op(a, lda, t, j, l);
        s += p, sa += fabsl(p);
      }
      long double ref = alpha * s + beta * c0[i + (long)j * n];
      double r = (double)(fabsl(c[i + (long)j * n] - ref) / ((sa + fabs(beta * c0[i + (long)j * n])) * (k + 2) * EPS + 1e-300));
      if (!(r <= ratio)) ratio = r;
    }
  char name[128];
  snprintf(name, sizeof name, "dsyrk_%c%c_n%d_k%d", uplo, t, n, k);
  record(name, ratio);
  free(a), free(c), free(c0);
}

static void check_symv(int n) {
  double *a = spd(n), *x = rmat(n, 1), *y = calloc(n, sizeof(double));
  double alpha = 1, beta = 0, ratio = 0;
  int one = 1;
  dsymv_("L", &n, &alpha, a, &n, x, &one, &beta, y, &one);
  for (int i = 0; i < n; i++) {
    long double s = 0, sa = 0;
    for (int l = 0; l < n; l++) {
      long double p = (long double)a[i + (long)l * n] * x[l];
      s += p, sa += fabsl(p);
    }
    double r = (double)(fabsl(y[i] - s) / (sa * (n + 2) * EPS + 1e-300));
    if (!(r <= ratio)) ratio = r;
  }
  char name[128];
  snprintf(name, sizeof name, "dsymv_L_n%d", n);
  record(name, ratio);
  free(a), free(x), free(y);
}

static void check_dot(int n) {
  double *x = rmat(n, 1), *y = rmat(n, 1);
  int one = 1;
  double d = ddot_(&n, x, &one, y, &one);
  long double s = 0, sa = 0;
  for (int i = 0; i < n; i++)
    s += (long double)x[i] * y[i], sa += fabsl((long double)x[i] * y[i]);
  char name[128];
  snprintf(name, sizeof name, "ddot_n%d", n);
  record(name, (double)(fabsl(d - s) / (sa * (n + 2) * EPS + 1e-300)));
  free(x), free(y);
}

/* residual of op(A) X = alpha B (side L) or X op(A) = alpha B (side R), A triangular, well conditioned */
static void check_trsm(int m, int n, char side, char uplo, char t) {
  int na = side == 'L' ? m : n;
  double* a = rmat(na, na);
  for (int j = 0; j < na; j++)
    for (int i = 0; i < na; i++) {
      if ((uplo == 'L' && i < j) || (uplo == 'U' && i > j)) a[i + (long)j * na] = 0;
      if (i == j) a[i + (long)j * na] = 2.0 + fabs(a[i + (long)j * na]);
      else a[i + (long)j * na] /= na;
    }
  double *b = rmat(m, n), *x = malloc(sizeof(double) * (size_t)m * n);
  memcpy(x, b, sizeof(double) * (size_t)m * n);
  double alpha = 1.5, ratio = 0;
  dtrsm_(&side, &uplo, &t, "N", &m, &n, &alpha, a, &na, x, &m);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      long double s = 0, sa = 0;
      for (int l = 0; l < na; l++) {
        long double p = side == 'L' ? (long double)op(a, na, t, i, l) * x[l + (long)j * m]
                                    : (long double)x[i + (long)l * m] * op(a, na, t, l, j);
        s += p, sa += fabsl(p);
      }
      double r = (double)(fabsl(s - alpha * b[i + (long)j * m]) / ((sa + fabs(alpha * b[i + (long)j * m])) * (na + 2) * EPS * 4 + 1e-300));
      if (!(r <= ratio)) ratio = r;
    }
  char name[128];
  snprintf(name, sizeof name, "dtrsm_%c%c%c_m%d_n%d", side, uplo, t, m, n);
  record(name, ratio);
  free(a), free(b), free(x);
}

static void check_potrf(int n, char uplo) {
  double *a = spd(n), *l = malloc(sizeof(double) * (size_t)n * n);
  memcpy(l, a, sizeof(double) * (size_t)n * n);
  blasint info = 0;
  dpotrf_(&uplo, &n, l, &n, &info);
  double ratio = 0;
  if (info != 0) ratio = INFINITY;
  else
    for (int j = 0; j < n; j++)
      for (int i = j; i < n; i++) { /* (L L^T)(i,j), or (U^T U)(j,i) */
        long double s = 0, sa = 0;
        for (int k = 0; k <= j; k++) {
          long double p = uplo == 'L' ? (long double)l[i + (long)k * n] * l[j + (long)k * n]
                                      : (long double)l[k + (long)i * n] * l[k + (long)j * n];
          s += p, sa += fabsl(p);
        }
        double r = (double)(fabsl(s - a[i + (long)j * n]) / (sa * (n + 2) * EPS * 4 + 1e-300));
        if (!(r <= ratio)) ratio = r;
      }
  char name[128];
  snprintf(name, sizeof name, "dpotrf_%c_n%d_info%d", uplo, n, (int)info);
  record(name, ratio);
  free(a), free(l);
}

static void check_gesv(int n, int nrhs) {
  double *a = spd(n), *lu = malloc(sizeof(double) * (size_t)n * n), *b = rmat(n, nrhs), *x = malloc(sizeof(double) * (size_t)n * nrhs);
  for (int i = 0; i < n * n; i++)
    a[i] += 0.1 * urand(); /* not symmetric */
  memcpy(lu, a, sizeof(double) * (size_t)n * n);
  memcpy(x, b, sizeof(double) * (size_t)n * nrhs);
  blasint* ipiv = malloc(sizeof(blasint) * n);
  blasint info = 0;
  dgesv_(&n, &nrhs, lu, &n, ipiv, x, &n, &info);
  double ratio = 0;
  if (info != 0) ratio = INFINITY;
  else
    for (int j = 0; j < nrhs; j++)
      for (int i = 0; i < n; i++) {
        long double s = 0, sa = 0;
        for (int l = 0; l < n; l++) {
          long double p = (long double)a[i + (long)l * n] * x[l + (long)j * n];
          s += p, sa += fabsl(p);
        }
        /* backward-stable solve: residual ~ n eps |A||x|, with some growth margin */
        double r = (double)(fabsl(s - b[i + (long)j * n]) / ((sa + fabs(b[i + (long)j * n])) * (n + 2) * EPS * 16 + 1e-300));
        if (!(r <= ratio)) ratio = r;
      }
  char name[128];
  snprintf(name, sizeof name, "dgesv_n%d_nrhs%d_info%d", n, nrhs, (int)info);
  record(name, ratio);
  free(a), free(lu), free(b), free(x), free(ipiv);
}

/* LAPACK routines armadillo / libKriging call besides the ones above */
extern void dpotrs_(const char*, const blasint*, const blasint*, const double*, const blasint*, double*, const blasint*,
                    blasint*);
extern void dtrtrs_(const char*, const char*, const char*, const blasint*, const blasint*, const double*,
                    const blasint*, double*, const blasint*, blasint*);
extern void dposv_(const char*, const blasint*, const blasint*, double*, const blasint*, double*, const blasint*,
                   blasint*);
extern void dsyevd_(const char*, const char*, const blasint*, double*, const blasint*, double*, double*,
                    const blasint*, blasint*, const blasint*, blasint*);
extern void dgeqrf_(const blasint*, const blasint*, double*, const blasint*, double*, double*, const blasint*,
                    blasint*);
extern void dorgqr_(const blasint*, const blasint*, const blasint*, double*, const blasint*, const double*, double*,
                    const blasint*, blasint*);
extern void dgesdd_(const char*, const blasint*, const blasint*, double*, const blasint*, double*, double*,
                    const blasint*, double*, const blasint*, double*, const blasint*, blasint*, blasint*);
extern void dtrcon_(const char*, const char*, const char*, const blasint*, const double*, const blasint*, double*,
                    double*, blasint*, blasint*);
extern void dpocon_(const char*, const blasint*, const double*, const blasint*, const double*, double*, double*,
                    blasint*, blasint*);

/* max_ij |(A X)_ij - B_ij| / ((|A||X| + |B|)_ij n eps 16) for n x n A, n x k X, B */
static double residual(int n, int k, const double* a, const double* x, const double* b) {
  double ratio = 0;
  for (int j = 0; j < k; j++)
    for (int i = 0; i < n; i++) {
      long double s = 0, sa = 0;
      for (int l = 0; l < n; l++) {
        long double p = (long double)a[i + (long)l * n] * x[l + (long)j * n];
        s += p, sa += fabsl(p);
      }
      double r = (double)(fabsl(s - b[i + (long)j * n]) / ((sa + fabs(b[i + (long)j * n])) * (n + 2) * EPS * 16 + 1e-300));
      if (!(r <= ratio)) ratio = r;
    }
  return ratio;
}

static void check_lapack_spd(int n, int nrhs) {
  char name[128];
  double *a = spd(n), *f = malloc(sizeof(double) * (size_t)n * n), *b = rmat(n, nrhs), *x = malloc(sizeof(double) * (size_t)n * nrhs);
  blasint info = 0, one = 1;
  /* dposv */
  memcpy(f, a, sizeof(double) * (size_t)n * n), memcpy(x, b, sizeof(double) * (size_t)n * nrhs);
  dposv_("L", &n, &nrhs, f, &n, x, &n, &info);
  snprintf(name, sizeof name, "dposv_L_n%d_nrhs%d_info%d", n, nrhs, (int)info);
  record(name, info ? INFINITY : residual(n, nrhs, a, x, b));
  /* dpotrs on the dpotrf factor */
  memcpy(f, a, sizeof(double) * (size_t)n * n), memcpy(x, b, sizeof(double) * (size_t)n * nrhs);
  dpotrf_("U", &n, f, &n, &info);
  if (!info) dpotrs_("U", &n, &nrhs, f, &n, x, &n, &info);
  snprintf(name, sizeof name, "dpotrs_U_n%d_nrhs%d_info%d", n, nrhs, (int)info);
  record(name, info ? INFINITY : residual(n, nrhs, a, x, b));
  /* dpocon (from the U factor) and dtrcon (on L = U^T) against the exact 1-norm rcond */
  double anorm = 0;
  for (int j = 0; j < n; j++) {
    double s = 0;
    for (int i = 0; i < n; i++)
      s += fabs(a[i + (long)j * n]);
    if (s > anorm) anorm = s;
  }
  double *inv = malloc(sizeof(double) * (size_t)n * n), *e = calloc((size_t)n * n, sizeof(double));
  for (int i = 0; i < n; i++)
    e[i + (long)i * n] = 1;
  memcpy(inv, e, sizeof(double) * (size_t)n * n);
  if (!info) dpotrs_("U", &n, &n, f, &n, inv, &n, &info);
  double inorm = 0;
  for (int j = 0; j < n; j++) {
    double s = 0;
    for (int i = 0; i < n; i++)
      s += fabs(inv[i + (long)j * n]);
    if (s > inorm) inorm = s;
  }
  double rc_exact = 1.0 / (anorm * inorm), rc = -1, *work = malloc(sizeof(double) * 3 * (size_t)n);
  blasint* iwork = malloc(sizeof(blasint) * (size_t)n);
  if (!info) dpocon_("U", &n, f, &n, &anorm, &rc, work, iwork, &info);
  /* LAPACK underestimates ||A^-1||_1: the estimate is >= the exact rcond, within a factor ~n */
  double rr = rc / rc_exact;
  snprintf(name, sizeof name, "dpocon_U_n%d", n);
  record(name, (info || !(rr > 1.0 - 1e-6 && rr < 10.0 * n)) ? INFINITY : 0.0);
  if (info || !(rr > 1.0 - 1e-6 && rr < 10.0 * n)) printf("  dpocon n=%d rcond %.3g exact %.3g\n", n, rc, rc_exact);
  if (!info) dtrcon_("1", "U", "N", &n, f, &n, &rc, work, iwork, &info);
  snprintf(name, sizeof name, "dtrcon_1UN_n%d", n);
  record(name, (info || !(rc > 0 && rc <= 1.0 + 1e-12)) ? INFINITY : 0.0);
  /* dtrtrs with the factor: U^T U x = b */
  memcpy(x, b, sizeof(double) * (size_t)n * nrhs);
  if (!info) dtrtrs_("U", "T", "N", &n, &nrhs, f, &n, x, &n, &info);
  if (!info) dtrtrs_("U", "N", "N", &n, &nrhs, f, &n, x, &n, &info);
  snprintf(name, sizeof name, "dtrtrs_U_TN_n%d_nrhs%d_info%d", n, nrhs, (int)info);
  record(name, info ? INFINITY : residual(n, nrhs, a, x, b));
  /* dsyevd: A V = V diag(w), V^T V = I */
  memcpy(f, a, sizeof(double) * (size_t)n * n);
  double* w = malloc(sizeof(double) * (size_t)n);
  blasint lwork = -1, liwork = -1, iw0 = 0;
  double w0 = 0;
  dsyevd_("V", "L", &n, f, &n, w, &w0, &lwork, &iw0, &liwork, &info);
  lwork = (blasint)w0, liwork = iw0;
  double* wk = malloc(sizeof(double) * (size_t)(lwork > 1 ? lwork : 1));
  blasint* iwk = malloc(sizeof(blasint) * (size_t)(liwork > 1 ? liwork : 1));
  dsyevd_("V", "L", &n, f, &n, w, wk, &lwork, iwk, &liwork, &info);
  double ratio = 0;
  if (info) ratio = INFINITY;
  else
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++) {
        long double s = 0, sa = 0, q = 0;
        for (int l = 0; l < n; l++) {
          long double p = (long double)a[i + (long)l * n] * f[l + (long)j * n];
          s += p, sa += fabsl(p);
          q += (long double)f[l + (long)i * n] * f[l + (long)j * n];
        }
        double r1 = (double)(fabsl(s - w[j] * f[i + (long)j * n]) / ((sa + fabs(w[j] * f[i + (long)j * n])) * (n + 2) * EPS * 64 + 1e-300));
        double r2 = (double)(fabsl(q - (i == j)) / ((n + 2) * EPS * 64));
        if (!(r1 <= ratio)) ratio = r1;
        if (!(r2 <= ratio)) ratio = r2;
      }
  snprintf(name, sizeof name, "dsyevd_VL_n%d_info%d", n, (int)info);
  record(name, ratio);
  free(a), free(f), free(b), free(x), free(inv), free(e), free(work), free(iwork), free(w), free(wk), free(iwk);
  (void)one;
}

static void check_qr_svd(int m, int n) {
  char name[128];
  int k = m < n ? m : n;
  double *a = rmat(m, n), *f = malloc(sizeof(double) * (size_t)m * n), *tau = malloc(sizeof(double) * (size_t)k);
  blasint info = 0, lwork = -1;
  double w0 = 0;
  /* dgeqrf + dorgqr (economy, m >= n): Q R = A, Q^T Q = I */
  if (m >= n) {
    memcpy(f, a, sizeof(double) * (size_t)m * n);
    dgeqrf_(&m, &n, f, &m, tau, &w0, &lwork, &info);
    lwork = (blasint)w0;
    double* wk = malloc(sizeof(double) * (size_t)(lwork > 1 ? lwork : 1));
    dgeqrf_(&m, &n, f, &m, tau, wk, &lwork, &info);
    double* r = calloc((size_t)n * n, sizeof(double));
    for (int j = 0; j < n; j++)
      for (int i = 0; i <= j; i++)
        r[i + (long)j * n] = f[i + (long)j * m];
    lwork = -1;
    dorgqr_(&m, &n, &n, f, &m, tau, &w0, &lwork, &info);
    lwork = (blasint)w0;
    wk = realloc(wk, sizeof(double) * (size_t)(lwork > 1 ? lwork : 1));
    if (!info) dorgqr_(&m, &n, &n, f, &m, tau, wk, &lwork, &info);
    double ratio = info ? INFINITY : 0;
    for (int j = 0; j < n && !info; j++)
      for (int i = 0; i < m; i++) {
        long double s = 0, sa = 0;
        for (int l = 0; l <= j; l++) {
          long double p = (long double)f[i + (long)l * m] * r[l + (long)j * n];
          s += p, sa += fabsl(p);
        }
        double rr = (double)(fabsl(s - a[i + (long)j * m]) / ((sa + fabs(a[i + (long)j * m])) * (m + 2) * EPS * 64 + 1e-300));
        if (!(rr <= ratio)) ratio = rr;
      }
    snprintf(name, sizeof name, "dgeqrf_dorgqr_m%d_n%d_info%d", m, n, (int)info);
    record(name, ratio);
    free(wk), free(r);
  }
  /* dgesdd ("S"): U diag(s) V^T = A */
  memcpy(f, a, sizeof(double) * (size_t)m * n);
  double *s = malloc(sizeof(double) * (size_t)k), *u = malloc(sizeof(double) * (size_t)m * k), *vt = malloc(sizeof(double) * (size_t)k * n);
  blasint* iw = malloc(sizeof(blasint) * 8 * (size_t)k);
  lwork = -1;
  dgesdd_("S", &m, &n, f, &m, s, u, &m, vt, &k, &w0, &lwork, iw, &info);
  lwork = (blasint)w0;
  double* wk = malloc(sizeof(double) * (size_t)(lwork > 1 ? lwork : 1));
  dgesdd_("S", &m, &n, f, &m, s, u, &m, vt, &k, wk, &lwork, iw, &info);
  double ratio = info ? INFINITY : 0;
  for (int j = 0; j < n && !info; j++)
    for (int i = 0; i < m; i++) {
      long double t = 0, ta = 0;
      for (int l = 0; l < k; l++) {
        long double p = (long double)u[i + (long)l * m] * s[l] * vt[l + (long)j * k];
        t += p, ta += fabsl(p);
      }
      double rr = (double)(fabsl(t - a[i + (long)j * m]) / ((ta + fabs(a[i + (long)j * m])) * (m + n + 2) * EPS * 64 + 1e-300));
      if (!(rr <= ratio)) ratio = rr;
    }
  snprintf(name, sizeof name, "dgesdd_S_m%d_n%d_info%d", m, n, (int)info);
  record(name, ratio);
  free(a), free(f), free(tau), free(s), free(u), free(vt), free(iw), free(wk);
}

int main(void) {
  srand(429);
  const char* core = "unknown";
  int threads = -1;
  void* h = dlopen(NULL, RTLD_NOW);
  char* (*corename)(void) = h ? (char* (*)(void))dlsym(h, "openblas_get_corename") : NULL;
  char* (*config)(void) = h ? (char* (*)(void))dlsym(h, "openblas_get_config") : NULL;
  int (*nthreads)(void) = h ? (int (*)(void))dlsym(h, "openblas_get_num_threads") : NULL;
  if (corename) core = corename();
  if (nthreads) threads = nthreads();
  char other[256] = "";
  void (*mkl_version)(char*, int) = h ? (void (*)(char*, int))dlsym(h, "mkl_get_version_string") : NULL;
  int (*mkl_threads)(void) = h ? (int (*)(void))dlsym(h, "MKL_Get_Max_Threads") : NULL;
  char* (*bli_version)(void) = h ? (char* (*)(void))dlsym(h, "bli_info_get_version_str") : NULL;
  int (*bli_arch_id)(void) = h ? (int (*)(void))dlsym(h, "bli_arch_query_id") : NULL;
  const char* (*bli_arch_str)(int) = h ? (const char* (*)(int))dlsym(h, "bli_arch_string") : NULL;
  if (mkl_version) {
    char v[198] = "";
    mkl_version(v, (int)sizeof v - 1);
    snprintf(other, sizeof other, "MKL %s", v);
    core = "mkl";
    if (mkl_threads) threads = mkl_threads();
  } else if (bli_version) {
    const char* arch = (bli_arch_id && bli_arch_str) ? bli_arch_str(bli_arch_id()) : "?";
    snprintf(other, sizeof other, "BLIS %s arch %s", bli_version(), arch);
    static char core_blis[64];
    snprintf(core_blis, sizeof core_blis, "blis-%s", arch);
    core = core_blis;
  }
  printf("BLAS: %s | core: %s | threads: %d\n", config ? config() : (other[0] ? other : "not OpenBLAS"), core, threads);

  static const int sq[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 96, 100, 127, 128, 129, 200, 256, 257};
  static const int shapes[][3] = {{100, 1, 100}, {1, 100, 100}, {100, 100, 1}, {37, 211, 19}, {211, 37, 19}, {19, 37, 211},
                                  {500, 4, 500}, {4, 500, 500}, {500, 500, 4}, {300, 300, 300}, {450, 10, 3}, {10, 450, 3}};
  const char tr[] = {'N', 'T'};
  for (unsigned s = 0; s < sizeof sq / sizeof *sq; s++)
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++) {
        check_gemm(sq[s], sq[s], sq[s], tr[a], tr[b], 0.0);
        check_gemm(sq[s], sq[s], sq[s], tr[a], tr[b], 1.0);
      }
  for (unsigned s = 0; s < sizeof shapes / sizeof *shapes; s++)
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++)
        check_gemm(shapes[s][0], shapes[s][1], shapes[s][2], tr[a], tr[b], 0.0);

  static const int vs[] = {1, 3, 8, 17, 64, 100, 257, 1000};
  for (unsigned s = 0; s < sizeof vs / sizeof *vs; s++) {
    check_dot(vs[s]);
    check_gemv(vs[s], vs[(s + 3) % 8], 'N');
    check_gemv(vs[s], vs[(s + 3) % 8], 'T');
  }
  static const int ls[] = {1, 2, 5, 8, 16, 17, 33, 64, 100, 128, 200, 300, 450};
  for (unsigned s = 0; s < sizeof ls / sizeof *ls; s++) {
    int n = ls[s];
    check_symv(n);
    check_syrk(n, n, 'L', 'N'), check_syrk(n, n, 'U', 'T'), check_syrk(n, 7, 'L', 'T');
    for (int u = 0; u < 2; u++)
      for (int t = 0; t < 2; t++) {
        check_trsm(n, 13, 'L', "LU"[u], tr[t]);
        check_trsm(13, n, 'R', "LU"[u], tr[t]);
      }
    check_potrf(n, 'L'), check_potrf(n, 'U');
    check_gesv(n, 1), check_gesv(n, 9);
    check_lapack_spd(n, 1), check_lapack_spd(n, 9);
  }
  static const int qs[][2] = {{1, 1}, {5, 3}, {17, 17}, {64, 8}, {100, 30}, {200, 200}, {33, 65}, {450, 4}};
  for (unsigned s = 0; s < sizeof qs / sizeof *qs; s++)
    check_qr_svd(qs[s][0], qs[s][1]);

  printf("SUMMARY core=%s threads=%d cases=%d fails=%d worst=%s ratio=%.3g\n", core, threads, n_cases, n_fails, worst_case,
         worst_ratio);
  return 0;
}
