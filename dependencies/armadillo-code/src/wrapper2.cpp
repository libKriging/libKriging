// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


#include <climits>
#include <limits>
#include <complex>

#if (__cplusplus >= 201103L)
  #undef  ARMA_USE_CXX11
  #define ARMA_USE_CXX11
#endif

#include "armadillo_bits/config.hpp"

#undef  ARMA_USE_WRAPPER

#undef  ARMA_USE_FORTRAN_HIDDEN_ARGS
#define ARMA_USE_FORTRAN_HIDDEN_ARGS

#include "armadillo_bits/compiler_setup.hpp"
#include "armadillo_bits/typedef_elem.hpp"

namespace arma
{

#include "armadillo_bits/def_blas.hpp"
#include "armadillo_bits/def_lapack.hpp"
#include "armadillo_bits/def_arpack.hpp"


extern "C"
  {
  #if defined(ARMA_USE_BLAS)
    
    float arma_fortran_with_prefix(arma_sasum)(blas_int* n, const float* x, blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_sasum)(n, x, incx);
      }
    
    double arma_fortran_with_prefix(arma_dasum)(blas_int* n, const double* x, blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_dasum)(n, x, incx);
      }
    
    
    
    float arma_fortran_with_prefix(arma_snrm2)(blas_int* n, const float* x, blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_snrm2)(n, x, incx);
      }
    
    double arma_fortran_with_prefix(arma_dnrm2)(blas_int* n, const double* x, blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_dnrm2)(n, x, incx);
      }
    
    
    
    float arma_fortran_with_prefix(arma_sdot)(blas_int* n, const float*  x, blas_int* incx, const float*  y, blas_int* incy)
      {
      return arma_fortran_sans_prefix(arma_sdot)(n, x, incx, y, incy);
      }
    
    double arma_fortran_with_prefix(arma_ddot)(blas_int* n, const double* x, blas_int* incx, const double* y, blas_int* incy)
      {
      return arma_fortran_sans_prefix(arma_ddot)(n, x, incx, y, incy);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*  alpha, const float*  A, const blas_int* ldA, const float*  x, const blas_int* incx, const float*  beta, float*  y, const blas_int* incy, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_sgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy, transA_len);
      }
    
    void arma_fortran_with_prefix(arma_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double* alpha, const double* A, const blas_int* ldA, const double* x, const blas_int* incx, const double* beta, double* y, const blas_int* incy, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_dgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy, transA_len);
      }
    
    void arma_fortran_with_prefix(arma_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_cgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy, transA_len);
      }
    
    void arma_fortran_with_prefix(arma_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_zgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy, transA_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*  alpha, const float*  A, const blas_int* ldA, const float*  B, const blas_int* ldB, const float*  beta, float*  C, const blas_int* ldC, blas_len transA_len, blas_len transB_len)
      {
      arma_fortran_sans_prefix(arma_sgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, transA_len, transB_len);
      }
    
    void arma_fortran_with_prefix(arma_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* B, const blas_int* ldB, const double* beta, double* C, const blas_int* ldC, blas_len transA_len, blas_len transB_len)
      {
      arma_fortran_sans_prefix(arma_dgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, transA_len, transB_len);
      }
    
    void arma_fortran_with_prefix(arma_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC, blas_len transA_len, blas_len transB_len)
      {
      arma_fortran_sans_prefix(arma_cgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, transA_len, transB_len);
      }
    
    void arma_fortran_with_prefix(arma_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC, blas_len transA_len, blas_len transB_len)
      {
      arma_fortran_sans_prefix(arma_zgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, transA_len, transB_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_ssyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC, uplo_len, transA_len);
      }
    
    void arma_fortran_with_prefix(arma_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_dsyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC, uplo_len, transA_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cherk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const void* A, const blas_int* ldA, const  float* beta, void* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_cherk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC, uplo_len, transA_len);
      }
    
    void arma_fortran_with_prefix(arma_zherk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const void* A, const blas_int* ldA, const double* beta, void* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len)
      {
      arma_fortran_sans_prefix(arma_zherk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC, uplo_len, transA_len);
      }
    
  #endif
  
  
  
  #if defined(ARMA_USE_LAPACK)
    
    void arma_fortran_with_prefix(arma_sgetrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgetrf)(m, n, a, lda, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_dgetrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgetrf)(m, n, a, lda, ipiv, info);
      }

    void arma_fortran_with_prefix(arma_cgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgetrf)(m, n, a, lda, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_zgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgetrf)(m, n, a, lda, ipiv, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgetri)(blas_int* n,  float* a, blas_int* lda, blas_int* ipiv,  float* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgetri)(blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgetri)(blas_int* n,  void*  a, blas_int* lda, blas_int* ipiv,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgetri)(blas_int* n,  void*  a, blas_int* lda, blas_int* ipiv,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strtri)(char* uplo, char* diag, blas_int* n,  float* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_strtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_dtrtri)(char* uplo, char* diag, blas_int* n, double* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_dtrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_ctrtri)(char* uplo, char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_ctrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_ztrtri)(char* uplo, char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_ztrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyev)(char* jobz, char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_ssyev)(jobz, uplo, n, a, lda, w, work, lwork, info, jobz_len, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dsyev)(char* jobz, char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dsyev)(jobz, uplo, n, a, lda, w, work, lwork, info, jobz_len, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cheev)(char* jobz, char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info, jobz_len, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zheev)(char* jobz, char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info, jobz_len, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyevd)(char* jobz, char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_ssyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, jobz_len, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dsyevd)(char* jobz, char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, jobz_len, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cheevd)(char* jobz, char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info, jobz_len, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zheevd)(char* jobz, char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info, jobz_len, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeev)(char* jobvl, char* jobvr, blas_int* n,  float* a, blas_int* lda,  float* wr,  float* wi,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr,  float* work, blas_int* lwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_sgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info, jobvl_len, jobvr_len);
      }
    
    void arma_fortran_with_prefix(arma_dgeev)(char* jobvl, char* jobvr, blas_int* n, double* a, blas_int* lda, double* wr, double* wi, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, double* work, blas_int* lwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_dgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info, jobvl_len, jobvr_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgeev)(char* jobvl, char* jobvr, blas_int* n, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork,  float* rwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_cgeev)(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info, jobvl_len, jobvr_len);
      }
    
    void arma_fortran_with_prefix(arma_zgeev)(char* jobvl, char* jobvr, blas_int* n, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_zgeev)(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info, jobvl_len, jobvr_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeevx)(char* balanc, char* jobvl, char* jobvr, char* sense, blas_int* n,  float* a, blas_int* lda,  float* wr,  float* wi,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr, blas_int* ilo, blas_int* ihi,  float* scale,  float* abnrm,  float* rconde,  float* rcondv,  float* work, blas_int* lwork, blas_int* iwork, blas_int* info, blas_len balanc_len, blas_len jobvl_len, blas_len jobvr_len, blas_len sense_len)
      {
      arma_fortran_sans_prefix(arma_sgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info, balanc_len, jobvl_len, jobvr_len, sense_len);
      }
    
    void arma_fortran_with_prefix(arma_dgeevx)(char* balanc, char* jobvl, char* jobvr, char* sense, blas_int* n, double* a, blas_int* lda, double* wr, double* wi, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, double* work, blas_int* lwork, blas_int* iwork, blas_int* info, blas_len balanc_len, blas_len jobvl_len, blas_len jobvr_len, blas_len sense_len)
      {
      arma_fortran_sans_prefix(arma_dgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info, balanc_len, jobvl_len, jobvr_len, sense_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgeevx)(char* balanc, char* jobvl, char* jobvr, char* sense, blas_int* n, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, blas_int* ilo, blas_int* ihi,  float* scale,  float* abnrm,  float* rconde,  float* rcondv, void* work, blas_int* lwork,  float* rwork, blas_int* info, blas_len balanc_len, blas_len jobvl_len, blas_len jobvr_len, blas_len sense_len)
      {
      arma_fortran_sans_prefix(arma_cgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, rwork, info, balanc_len, jobvl_len, jobvr_len, sense_len);
      }
    
    void arma_fortran_with_prefix(arma_zgeevx)(char* balanc, char* jobvl, char* jobvr, char* sense, blas_int* n, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, void* work, blas_int* lwork, double* rwork, blas_int* info, blas_len balanc_len, blas_len jobvl_len, blas_len jobvr_len, blas_len sense_len)
      {
      arma_fortran_sans_prefix(arma_zgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, rwork, info, balanc_len, jobvl_len, jobvr_len, sense_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sggev)(char* jobvl, char* jobvr, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* alphar,  float* alphai,  float* beta,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr,  float* work, blas_int* lwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_sggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info, jobvl_len, jobvr_len);
      }
      
    void arma_fortran_with_prefix(arma_dggev)(char* jobvl, char* jobvr, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, double* alphar, double* alphai, double* beta, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, double* work, blas_int* lwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_dggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info, jobvl_len, jobvr_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cggev)(char* jobvl, char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork,  float* rwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_cggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info, jobvl_len, jobvr_len);
      }
    
    void arma_fortran_with_prefix(arma_zggev)(char* jobvl, char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info, blas_len jobvl_len, blas_len jobvr_len)
      {
      arma_fortran_sans_prefix(arma_zggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info, jobvl_len, jobvr_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spotrf)(char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_spotrf)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dpotrf)(char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dpotrf)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_cpotrf)(char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cpotrf)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zpotrf)(char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zpotrf)(uplo, n, a, lda, info, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spbtrf)(char* uplo, blas_int* n, blas_int* kd,  float* ab, blas_int* ldab, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_spbtrf)(uplo, n, kd, ab, ldab, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dpbtrf)(char* uplo, blas_int* n, blas_int* kd, double* ab, blas_int* ldab, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dpbtrf)(uplo, n, kd, ab, ldab, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_cpbtrf)(char* uplo, blas_int* n, blas_int* kd,   void* ab, blas_int* ldab, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cpbtrf)(uplo, n, kd, ab, ldab, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zpbtrf)(char* uplo, blas_int* n, blas_int* kd,   void* ab, blas_int* ldab, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zpbtrf)(uplo, n, kd, ab, ldab, info, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spotri)(char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_spotri)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dpotri)(char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dpotri)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_cpotri)(char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cpotri)(uplo, n, a, lda, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zpotri)(char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zpotri)(uplo, n, a, lda, info, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeqrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgeqrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sorgqr)(blas_int* m, blas_int* n, blas_int* k,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dorgqr)(blas_int* m, blas_int* n, blas_int* k, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesvd)(char* jobu, char* jobvt, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len)
      {
      arma_fortran_sans_prefix(arma_sgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info, jobu_len, jobvt_len);
      }
    
    void arma_fortran_with_prefix(arma_dgesvd)(char* jobu, char* jobvt, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len)
      {
      arma_fortran_sans_prefix(arma_dgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info, jobu_len, jobvt_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgesvd)(char* jobu, char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, float*  s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, float*  rwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len)
      {
      arma_fortran_sans_prefix(arma_cgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info, jobu_len, jobvt_len);
      }
    
    void arma_fortran_with_prefix(arma_zgesvd)(char* jobu, char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, double* s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, double* rwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len)
      {
      arma_fortran_sans_prefix(arma_zgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info, jobu_len, jobvt_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesdd)(char* jobz, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* iwork, blas_int* info, blas_len jobz_len)
      {
      arma_fortran_sans_prefix(arma_sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info, jobz_len);
      }
    
    void arma_fortran_with_prefix(arma_dgesdd)(char* jobz, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* iwork, blas_int* info, blas_len jobz_len)
      {
      arma_fortran_sans_prefix(arma_dgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info, jobz_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgesdd)(char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, float*  s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, float*  rwork, blas_int* iwork, blas_int* info, blas_len jobz_len)
      {
      arma_fortran_sans_prefix(arma_cgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info, jobz_len);
      }
    
    void arma_fortran_with_prefix(arma_zgesdd)(char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, double* s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info, blas_len jobz_len)
      {
      arma_fortran_sans_prefix(arma_zgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info, jobz_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesv)(blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgesv)(blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* af, blas_int* ldaf, blas_int* ipiv, char* equed,  float* r,  float* c,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_sgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_dgesvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* af, blas_int* ldaf, blas_int* ipiv, char* equed, double* r, double* c, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_dgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len, equed_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgesvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, char* equed,  float* r,  float* c, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_cgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_zgesvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, char* equed, double* r, double* c, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_zgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len, equed_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sposv)(char* uplo, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* b, blas_int* ldb, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_sposv)(uplo, n, nrhs, a, lda, b, ldb, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dposv)(char* uplo, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dposv)(uplo, n, nrhs, a, lda, b, ldb, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_cposv)(char* uplo, blas_int* n, blas_int* nrhs,   void* a, blas_int* lda,   void* b, blas_int* ldb, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cposv)(uplo, n, nrhs, a, lda, b, ldb, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zposv)(char* uplo, blas_int* n, blas_int* nrhs,   void* a, blas_int* lda,   void* b, blas_int* ldb, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zposv)(uplo, n, nrhs, a, lda, b, ldb, info, uplo_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sposvx)(char* fact, char* uplo, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* af, blas_int* ldaf, char* equed,  float* s,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len uplo_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_sposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, uplo_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_dposvx)(char* fact, char* uplo, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* af, blas_int* ldaf, char* equed, double* s, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len uplo_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_dposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, uplo_len, equed_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cposvx)(char* fact, char* uplo, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, char* equed,  float* s, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info, blas_len fact_len, blas_len uplo_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_cposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, uplo_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_zposvx)(char* fact, char* uplo, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, char* equed, double* s, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info, blas_len fact_len, blas_len uplo_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_zposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, uplo_len, equed_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgels)(char* trans, blas_int* m, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, float*  b, blas_int* ldb, float*  work, blas_int* lwork, blas_int* info, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_sgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info, trans_len);
      }
    
    void arma_fortran_with_prefix(arma_dgels)(char* trans, blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* work, blas_int* lwork, blas_int* info, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_dgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info, trans_len);
      }
    
    void arma_fortran_with_prefix(arma_cgels)(char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_cgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info, trans_len);
      }
    
    void arma_fortran_with_prefix(arma_zgels)(char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_zgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info, trans_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgelsd)(blas_int* m, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank,  float* work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, double* work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank, void* work, blas_int* lwork,  float* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_strtrs)(char* uplo, char* trans, char* diag, blas_int* n, blas_int* nrhs, const float*  a, blas_int* lda, float*  b, blas_int* ldb, blas_int* info, blas_len uplo_len, blas_len trans_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_strtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, uplo_len, trans_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_dtrtrs)(char* uplo, char* trans, char* diag, blas_int* n, blas_int* nrhs, const double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info, blas_len uplo_len, blas_len trans_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_dtrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, uplo_len, trans_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_ctrtrs)(char* uplo, char* trans, char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info, blas_len uplo_len, blas_len trans_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_ctrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, uplo_len, trans_len, diag_len);
      }
    
    void arma_fortran_with_prefix(arma_ztrtrs)(char* uplo, char* trans, char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info, blas_len uplo_len, blas_len trans_len, blas_len diag_len)
      {
      arma_fortran_sans_prefix(arma_ztrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, uplo_len, trans_len, diag_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgbsv)(blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs,  float* ab, blas_int* ldab, blas_int* ipiv,  float* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbsv)(blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs, double* ab, blas_int* ldab, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbsv)(blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs,   void* ab, blas_int* ldab, blas_int* ipiv,   void* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbsv)(blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs,   void* ab, blas_int* ldab, blas_int* ipiv,   void* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgbsvx)(char* fact, char* trans, blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs,  float* ab, blas_int* ldab,  float* afb, blas_int* ldafb, blas_int* ipiv, char* equed,  float* r,  float* c,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_sgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_dgbsvx)(char* fact, char* trans, blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs, double* ab, blas_int* ldab, double* afb, blas_int* ldafb, blas_int* ipiv, char* equed, double* r, double* c, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_dgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len, equed_len);
      }
    
    
    void arma_fortran_with_prefix(arma_cgbsvx)(char* fact, char* trans, blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs, void* ab, blas_int* ldab, void* afb, blas_int* ldafb, blas_int* ipiv, char* equed,  float* r,  float* c, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_cgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len, equed_len);
      }
    
    void arma_fortran_with_prefix(arma_zgbsvx)(char* fact, char* trans, blas_int* n, blas_int* kl, blas_int* ku, blas_int* nrhs, void* ab, blas_int* ldab, void* afb, blas_int* ldafb, blas_int* ipiv, char* equed, double* r, double* c, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info, blas_len fact_len, blas_len trans_len, blas_len equed_len)
      {
      arma_fortran_sans_prefix(arma_zgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len, equed_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgtsv)(blas_int* n, blas_int* nrhs,  float* dl,  float* d,  float* du,  float* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgtsv)(blas_int* n, blas_int* nrhs, double* dl, double* d, double* du, double* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgtsv)(blas_int* n, blas_int* nrhs,   void* dl,   void* d,   void* du,   void* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgtsv)(blas_int* n, blas_int* nrhs,   void* dl,   void* d,   void* du,   void* b, blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgtsvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs,  float* dl,  float* d,  float* du,  float* dlf,  float* df,  float* duf,  float* du2, blas_int* ipiv,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_sgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len);
      }
    
    void arma_fortran_with_prefix(arma_dgtsvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, double* dl, double* d, double* du, double* dlf, double* df, double* duf, double* du2, blas_int* ipiv, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info, blas_len fact_len, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_dgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info, fact_len, trans_len);
      }
    
    
    void arma_fortran_with_prefix(arma_cgtsvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, void* dl, void* d, void* du, void* dlf, void* df, void* duf, void* du2, blas_int* ipiv, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info, blas_len fact_len, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_cgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len);
      }
    
    void arma_fortran_with_prefix(arma_zgtsvx)(char* fact, char* trans, blas_int* n, blas_int* nrhs, void* dl, void* d, void* du, void* dlf, void* df, void* duf, void* du2, blas_int* ipiv, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info, blas_len fact_len, blas_len trans_len)
      {
      arma_fortran_sans_prefix(arma_zgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info, fact_len, trans_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgees)(char* jobvs, char* sort, void* select, blas_int* n, float*  a, blas_int* lda, blas_int* sdim, float*  wr, float*  wi, float*  vs, blas_int* ldvs, float*  work, blas_int* lwork, blas_int* bwork, blas_int* info, blas_len jobvs_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_sgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info, jobvs_len, sort_len);
      }
      
    void arma_fortran_with_prefix(arma_dgees)(char* jobvs, char* sort, void* select, blas_int* n, double* a, blas_int* lda, blas_int* sdim, double* wr, double* wi, double* vs, blas_int* ldvs, double* work, blas_int* lwork, blas_int* bwork, blas_int* info, blas_len jobvs_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_dgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info, jobvs_len, sort_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cgees)(char* jobvs, char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, float*  rwork, blas_int* bwork, blas_int* info, blas_len jobvs_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_cgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info, jobvs_len, sort_len);
      }
    
    void arma_fortran_with_prefix(arma_zgees)(char* jobvs, char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, double* rwork, blas_int* bwork, blas_int* info, blas_len jobvs_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_zgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info, jobvs_len, sort_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strsyl)(char* transa, char* transb, blas_int* isgn, blas_int* m, blas_int* n, const float*  a, blas_int* lda, const float*  b, blas_int* ldb, float*  c, blas_int* ldc, float*  scale, blas_int* info, blas_len transa_len, blas_len transb_len)
      {
      arma_fortran_sans_prefix(arma_strsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info, transa_len, transb_len);
      }
    
    void arma_fortran_with_prefix(arma_dtrsyl)(char* transa, char* transb, blas_int* isgn, blas_int* m, blas_int* n, const double* a, blas_int* lda, const double* b, blas_int* ldb, double* c, blas_int* ldc, double* scale, blas_int* info, blas_len transa_len, blas_len transb_len)
      {
      arma_fortran_sans_prefix(arma_dtrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info, transa_len, transb_len);
      }
    
    void arma_fortran_with_prefix(arma_ctrsyl)(char* transa, char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, float*  scale, blas_int* info, blas_len transa_len, blas_len transb_len)
      {
      arma_fortran_sans_prefix(arma_ctrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info, transa_len, transb_len);
      }
    
    void arma_fortran_with_prefix(arma_ztrsyl)(char* transa, char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, double* scale, blas_int* info, blas_len transa_len, blas_len transb_len)
      {
      arma_fortran_sans_prefix(arma_ztrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info, transa_len, transb_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgges)(char* jobvsl, char* jobvsr, char* sort, void* selctg, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb, blas_int* sdim,  float* alphar,  float* alphai,  float* beta,  float* vsl, blas_int* ldvsl,  float* vsr, blas_int* ldvsr,  float* work, blas_int* lwork,  float* bwork, blas_int* info, blas_len jobvsl_len, blas_len jobvsr_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_sgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info, jobvsl_len, jobvsr_len, sort_len);
      }
    
    void arma_fortran_with_prefix(arma_dgges)(char* jobvsl, char* jobvsr, char* sort, void* selctg, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* sdim, double* alphar, double* alphai, double* beta, double* vsl, blas_int* ldvsl, double* vsr, blas_int* ldvsr, double* work, blas_int* lwork, double* bwork, blas_int* info, blas_len jobvsl_len, blas_len jobvsr_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_dgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info, jobvsl_len, jobvsr_len, sort_len);
      }
    
    void arma_fortran_with_prefix(arma_cgges)(char* jobvsl, char* jobvsr, char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork,  float* rwork,  float* bwork, blas_int* info, blas_len jobvsl_len, blas_len jobvsr_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_cgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info, jobvsl_len, jobvsr_len, sort_len);
      }
    
    void arma_fortran_with_prefix(arma_zgges)(char* jobvsl, char* jobvsr, char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork, double* rwork, double* bwork, blas_int* info, blas_len jobvsl_len, blas_len jobvsr_len, blas_len sort_len)
      {
      arma_fortran_sans_prefix(arma_zgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info, jobvsl_len, jobvsr_len, sort_len);
      }
    
    
    
    
    float  arma_fortran_with_prefix(arma_slange)(char* norm, blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* work, blas_len norm_len)
      {
      return arma_fortran_sans_prefix(arma_slange)(norm, m, n, a, lda, work, norm_len);
      }
    
    double arma_fortran_with_prefix(arma_dlange)(char* norm, blas_int* m, blas_int* n, double* a, blas_int* lda, double* work, blas_len norm_len)
      {
      return arma_fortran_sans_prefix(arma_dlange)(norm, m, n, a, lda, work, norm_len);
      }
    
    float  arma_fortran_with_prefix(arma_clange)(char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda,  float* work, blas_len norm_len)
      {
      return arma_fortran_sans_prefix(arma_clange)(norm, m, n, a, lda, work, norm_len);
      }
    
    double arma_fortran_with_prefix(arma_zlange)(char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda, double* work, blas_len norm_len)
      {
      return arma_fortran_sans_prefix(arma_zlange)(norm, m, n, a, lda, work, norm_len);
      }
    
    
    
    
    void arma_fortran_with_prefix(arma_sgecon)(char* norm, blas_int* n,  float* a, blas_int* lda,  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info, blas_len norm_len)
      {
      arma_fortran_sans_prefix(arma_sgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info, norm_len);
      }
    
    void arma_fortran_with_prefix(arma_dgecon)(char* norm, blas_int* n, double* a, blas_int* lda, double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info, blas_len norm_len)
      {
      arma_fortran_sans_prefix(arma_dgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info, norm_len);
      }
    
    void arma_fortran_with_prefix(arma_cgecon)(char* norm, blas_int* n, void* a, blas_int* lda,  float* anorm,  float* rcond, void* work,  float* rwork, blas_int* info, blas_len norm_len)
      {
      arma_fortran_sans_prefix(arma_cgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info, norm_len);
      }
    
    void arma_fortran_with_prefix(arma_zgecon)(char* norm, blas_int* n, void* a, blas_int* lda, double* anorm, double* rcond, void* work, double* rwork, blas_int* info, blas_len norm_len)
      {
      arma_fortran_sans_prefix(arma_zgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info, norm_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spocon)(char* uplo, blas_int* n,  float* a, blas_int* lda,  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_spocon)(uplo, n, a, lda, anorm, rcond, work, iwork, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_dpocon)(char* uplo, blas_int* n, double* a, blas_int* lda, double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_dpocon)(uplo, n, a, lda, anorm, rcond, work, iwork, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_cpocon)(char* uplo, blas_int* n, void* a, blas_int* lda,  float* anorm,  float* rcond, void* work,  float* rwork, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_cpocon)(uplo, n, a, lda, anorm, rcond, work, rwork, info, uplo_len);
      }
    
    void arma_fortran_with_prefix(arma_zpocon)(char* uplo, blas_int* n, void* a, blas_int* lda, double* anorm, double* rcond, void* work, double* rwork, blas_int* info, blas_len uplo_len)
      {
      arma_fortran_sans_prefix(arma_zpocon)(uplo, n, a, lda, anorm, rcond, work, rwork, info, uplo_len);
      }
    
    
    
    blas_int arma_fortran_with_prefix(arma_ilaenv)(blas_int* ispec, char* name, char* opts, blas_int* n1, blas_int* n2, blas_int* n3, blas_int* n4, blas_len name_len, blas_len opts_len)
      {
      return arma_fortran_sans_prefix(arma_ilaenv)(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_slahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, float*  h, blas_int* ldh, float*  wr, float*  wi, blas_int* iloz, blas_int* ihiz, float*  z, blas_int* ldz, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_slahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }
    
    void arma_fortran_with_prefix(arma_dlahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, double* h, blas_int* ldh, double* wr, double* wi, blas_int* iloz, blas_int* ihiz, double* z, blas_int* ldz, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dlahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sstedc)(char* compz, blas_int* n, float*  d, float*  e, float*  z, blas_int* ldz, float*  work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len compz_len)
      {
      arma_fortran_sans_prefix(arma_sstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info, compz_len);
      }
    
    void arma_fortran_with_prefix(arma_dstedc)(char* compz, blas_int* n, double* d, double* e, double* z, blas_int* ldz, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len compz_len)
      {
      arma_fortran_sans_prefix(arma_dstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info, compz_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strevc)(char* side, char* howmny, blas_int* select, blas_int* n, float*  t, blas_int* ldt, float*  vl, blas_int* ldvl, float*  vr, blas_int* ldvr, blas_int* mm, blas_int* m, float*  work, blas_int* info, blas_len side_len, blas_len howmny_len)
      {
      arma_fortran_sans_prefix(arma_strevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info, side_len, howmny_len);
      }
    
    void arma_fortran_with_prefix(arma_dtrevc)(char* side, char* howmny, blas_int* select, blas_int* n, double* t, blas_int* ldt, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, blas_int* mm, blas_int* m, double* work, blas_int* info, blas_len side_len, blas_len howmny_len)
      {
      arma_fortran_sans_prefix(arma_dtrevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info, side_len, howmny_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_slarnv)(blas_int* idist, blas_int* iseed, blas_int* n, float*  x)
      {
      arma_fortran_sans_prefix(arma_slarnv)(idist, iseed, n, x);
      }
    
    void arma_fortran_with_prefix(arma_dlarnv)(blas_int* idist, blas_int* iseed, blas_int* n, double* x)
      {
      arma_fortran_sans_prefix(arma_dlarnv)(idist, iseed, n, x);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgehrd)(blas_int* n, blas_int* ilo, blas_int* ihi, float*  a, blas_int* lda, float*  tao, float*  work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgehrd)(blas_int* n, blas_int* ilo, blas_int* ihi, double* a, blas_int* lda, double* tao, double* work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgehrd)(blas_int* n, blas_int* ilo, blas_int* ihi, void*   a, blas_int* lda, void*   tao, void*   work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgehrd)(blas_int* n, blas_int* ilo, blas_int* ihi, void*   a, blas_int* lda, void*   tao, void*   work, blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
  #endif
  
  
  
  #if defined(ARMA_USE_ARPACK)
    
    void arma_fortran_with_prefix(arma_snaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_snaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_dnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_dnaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_cnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, float* rwork, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_cnaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_znaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, double* rwork, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_znaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info, bmat_len, which_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sneupd)(blas_int* rvec, char* howmny, blas_int* select, float* dr, float* di, float* z, blas_int* ldz, float* sigmar, float* sigmai, float* workev, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_sneupd)(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, howmny_len, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_dneupd)(blas_int* rvec, char* howmny, blas_int* select, double* dr, double* di, double* z, blas_int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_dneupd)(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, howmny_len, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_cneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d, void* z, blas_int* ldz, void* sigma, void* workev, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, float* rwork, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_cneupd)(rvec, howmny, select, d, z, ldz, sigma, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info, howmny_len, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_zneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d, void* z, blas_int* ldz, void* sigma, void* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, double* rwork, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_zneupd)(rvec, howmny, select, d, z, ldz, sigma, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info, howmny_len, bmat_len, which_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_ssaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_dsaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_dsaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, bmat_len, which_len);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sseupd)(blas_int* rvec, char* howmny, blas_int* select, float* d, float* z, blas_int* ldz, float* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_sseupd)(rvec, howmny, select, d, z, ldz, sigma, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, howmny_len, bmat_len, which_len);
      }
    
    void arma_fortran_with_prefix(arma_dseupd)(blas_int* rvec, char* howmny, blas_int* select, double* d, double* z, blas_int* ldz, double* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info, blas_len howmny_len, blas_len bmat_len, blas_len which_len)
      {
      arma_fortran_sans_prefix(arma_dseupd)(rvec, howmny, select, d, z, ldz, sigma, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, howmny_len, bmat_len, which_len);
      }
    
  #endif
  }  // end of extern "C"


}  // end of namespace arma
