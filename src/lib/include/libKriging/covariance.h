
#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

//===============================================================================
// unit used for computing covariances.
// Covariances act on Points that are auto-rescaled in order to save computation time.
// The unit aso handles Point type.
// classes:
// PointsStorage, CorrelationFunction, CovarianceParameters, Covariance, Points
//===============================================================================
//
// e.g. typical use:
//   CovarianceParameters covParams(dimension, lengthscales, variance, covFamilyString);
//   Covariance covariance(covParams);
//   Points pointsX(matrixX, covParams)
//   covariance.fillCorrMatrix(K, pointsX); // fill K with correlations matrix of X

#include <cmath> // exp, pow, sqrt... C++11 exp2
#include "common.h"
#include "messages.h"

//========================================================================== CHOSEN_STORAGE
// choice of the storage for Points:
// important because covariance calculation is one of the most expensive part in nested Kriging algo.
// storage may affect performance (cache locality, false sharing, alignement), affect memory footprint
//         and change available optimized methods (simd, arma::dot, valarray * etc.)
// see comments at the end of the unit for inserting new storages. Available choices:
// 1: std::vector<double>, 2: CompactMatrix, 3: std::vector<arma::vec>, 4: std::vector<valarray>, 5: arma::mat

#define CHOSEN_STORAGE 4

//----------- STORAGE 1: use vector<double>
#if CHOSEN_STORAGE == 1
  #define MATRIX_STORAGE 0
  using PointsStorage = std::vector<std::vector<double> >;
  using WritablePoint = PointsStorage::value_type;
  using Point = const PointsStorage::value_type;

//----------- STORAGE 2: use CompactMatrix, allows aligned, optimised storage for simd instructions
#elif CHOSEN_STORAGE == 2
  #include "compactMatrix.h"
  #define MATRIX_STORAGE 1
  using PointsStorage = nestedKrig::CompactMatrix;
  using Point = nestedKrig::CompactMatrix::constRow_type;
  using WritablePoint = nestedKrig::CompactMatrix::writableRow_type;

//----------- STORAGE 3: use vector<arma::vec>, allows arma::dot for corr
#elif CHOSEN_STORAGE == 3
  #define MATRIX_STORAGE 0
  using PointsStorage = std::vector<arma::vec>;
  using Point = const PointsStorage::value_type;
  using WritablePoint = PointsStorage::value_type;

  //----------- STORAGE 4: use valarray, allows vector operations for distances, .sum()
#elif CHOSEN_STORAGE == 4
  #include <valarray>
  #define MATRIX_STORAGE 0
  using PointsStorage = std::vector<std::valarray<double> >;
  using Point = const PointsStorage::value_type;
  using WritablePoint = PointsStorage::value_type;

  //----------- STORAGE 5: use arma::mat, allows bounds control, compact storage, arma::dot
#elif CHOSEN_STORAGE == 5
  #define MATRIX_STORAGE 1
  using PointsStorage = arma::mat;
  using Point = const arma::subview_row<double>;
  using WritablePoint = arma::subview_row<double>;
#endif

//========================================================== Covariance header

// namespace nestedKrig {

using Double = long double;

//========================================================== Tiny nuggets
// with tinyNuggetOnDiag, correlation matrix diagonal becomes
// 1.0 + factor*(smallest nugget) = 1.0 + factor * 2.22045e-016
// => good results on the inversion stability of MatrixOfOnes + Diag(nugget) that can occur in practice
// => results are better if the factor is a power of 2 for singular matrices of size up to 2*factor
// for matrices of size <=512, with factor= 256, nugget=5.68434e-014,
// max error regular case=5.68434e-014, max error singular case=5.68434e-014
// almost same results when setting diag=1 and all distances increased by nugget (tinyNuggetOffDiag)
// almost same results when combining the tinyNuggetOnDiag and tinyNuggetOffDiag
// cf. unit appendix_nuggetAnalysis.h

constexpr double tinyNuggetOnDiag = 256 * std::numeric_limits<double>::epsilon(); // 5.684...e-014
constexpr double tinyNuggetOffDiag = 0.0;

//========================================================== ApproximationTools
// Small utility functions to approximate exponential by a PSD function
//  . N is the order of the approximation
//  . raiseToPower_TwoPower<N>(x) computes x^(2^N), unrolled loop at compilation time
//  . exponentialOfMinus(x) approximates exp(-x) for positive values of x
//    it computes (1+x/m)^m where m = 2^N
//  . available for constexpr
//
//    furthermore expOfMinus(-x) and expOfMinus(-x^2) are positive semi-definite (PSD)
//    thus valid covariance functions for x Euclidean or Manhattan distance
//    thus this approx can be used, e.g., for Gaussian or Exp covariance kernel

struct ApproximationTools {
  template <int N>
  static inline constexpr double raiseToPower_TwoPower(double x) noexcept {
    return raiseToPower_TwoPower<N-1>(x * x);
  }

  template <int N>
  static inline constexpr double oneOver_TwoPower() noexcept {
    // returns 1.0/(two power N), limited to 32, but works up to order 64
    static_assert(N<32, "approx order should be less than 32");
    return 1.0/(1ULL<<N);
  }

  template <int N >
  static inline constexpr double exponentialOfMinus(double x) noexcept {
    return raiseToPower_TwoPower<N>( 1/(1 + x * oneOver_TwoPower<N>()) );
    // as x is a double, divisions are not integer divisions
  }
};

template <>
inline constexpr double ApproximationTools::raiseToPower_TwoPower<0>(double x) noexcept {
  return x;
}

//========================================================== Constants
//here use of constants because they are usually not constexpr in <cmath>
struct Math {
  static constexpr Double ln2          = 0.69314718055994530941723212145817656807550013436025L;
  static constexpr Double sqrt2        = 1.41421356237309504880168872420969807856967187537694L;
  static constexpr Double sqrt3        = 1.73205080756887729352744634150587236694280525381038L;
  static constexpr Double sqrt5        = 2.23606797749978969640917366873127623544061835961152L;
  static constexpr Double inv_ln2      = 1.0L/ln2;
};

//========================================================== ExponentialAlgorithm

struct Expo {
  inline static double expOfMinus(double x) noexcept { return std::exp(-x); }
  template <typename T>
  inline static constexpr T scale(const T x) noexcept { return x; };
  template <typename T>
  inline static constexpr T unscale(const T x) noexcept { return x; };
};

struct ExpoBase2  {
  //exp2(s)=2^{-s}, slightly faster than exp()
  //hence scale=logl(2.0L) to be applied to values within expo
  inline static double expOfMinus(double x) noexcept { return std::exp2(-x); }
  template <typename T>
  inline static constexpr T scale(const T x) noexcept { return x*Math::inv_ln2; };
  template <typename T>
  inline static constexpr T unscale(const T x) noexcept { return x*Math::ln2; };
};

template <int N>
struct ExpoApprox {
  inline static double expOfMinus(double x) noexcept { return ApproximationTools::exponentialOfMinus<N>(x); }
  template <typename T>
  inline static constexpr T scale(const T x) noexcept { return x; };
  template <typename T>
  inline static constexpr T unscale(const T x) noexcept { return x; };
};

//========================================================== CorrelationFunction
// Correlation functions
// to be applied to a data which has been rescaled (PointsStorage)
// (in order to improve performance, lengthscales are set to 1 for rescaled data)

class CorrelationFunction {
public:
  const PointDimension d;

  double norm1(const Point& x1, const Point& x2) const noexcept {
    double s = tinyNuggetOffDiag;
    // #pragma omp simd reduction (+:s) aligned(x1, x2:32)
    for (PointDimension k = 0; k < d; ++k) s += std::fabs(x1[k] - x2[k]);
    return s;
  }
  double norm2square(const Point& x1, const Point& x2) const noexcept {
    double s = tinyNuggetOffDiag;
    // #pragma omp simd reduction (+:s) aligned(x1, x2:32)
    for (PointDimension k = 0; k < d; ++k) {
      double t = x1[k] - x2[k];
      s += t*t;
    }
    return s;
  }
  double norm2(const Point& x1, const Point& x2) const noexcept {
    return std::sqrt(norm2square(x1,x2));
  }

public:
  CorrelationFunction(const PointDimension d) : d(d)  {  }

  virtual double corr(const Point& x1,const Point& x2) const noexcept =0;
  virtual Double scaling_factor() const =0;
  virtual ~CorrelationFunction(){}
};

//-------------- White Noise
class CorrWhiteNoise : public CorrelationFunction {
public:
  CorrWhiteNoise(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1,const Point& x2) const noexcept override {
    constexpr double treshold = tinyNuggetOffDiag + 256 * std::numeric_limits<double>::epsilon();
    double s = norm1(x1,x2);
    if (s < treshold) return 1.0;
    else return 0.0;
  }

  virtual Double scaling_factor() const override {
    return 1.0L;
  }
};

//-------------- Gauss
template <typename ExpoAlgo>
class CorrGauss : public CorrelationFunction {

public:

  CorrGauss(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    return ExpoAlgo::expOfMinus(norm2square(x1,x2));
  }

  virtual Double scaling_factor() const override {
    using namespace std; //because sqrtl should be in std, but usually is not
    return sqrtl(ExpoAlgo::scale(2.0L))/2.0L;
  }
};

//-------------- Rational 2
class CorrRational2 : public CorrelationFunction {
  static constexpr Double sqrt2OverTwo = Math::sqrt2/2.0L;

public:
  CorrRational2(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  double corr(const Point& x1, const Point& x2) const noexcept {
    return 1.0/(1.0+norm2square(x1,x2));
  }

  virtual Double scaling_factor() const {
    return sqrt2OverTwo;
  }
};
//-------------- Rational 1
class CorrRational1 : public CorrelationFunction {

public:
  CorrRational1(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  double corr(const Point& x1, const Point& x2) const noexcept {
    return 1.0/(1.0+norm1(x1,x2));
  }

  virtual Double scaling_factor() const {
    return 1.0L;
  }
};

//-------------- exp
template <typename ExpoAlgo>
class CorrExp : public CorrelationFunction {
public:
  CorrExp(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
      return ExpoAlgo::expOfMinus(norm1(x1,x2));
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(1.0L);
  }
};
//-------------- exp radial, draft not tested yet
template <typename ExpoAlgo>
class CorrExpRadial : public CorrelationFunction {
public:
  CorrExpRadial(const PointDimension d) :
  CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    return ExpoAlgo::expOfMinus(norm2(x1,x2));
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(1.0L);
  }
};

//-------------- Matern32
template <typename ExpoAlgo>
class CorrMatern32 : public CorrelationFunction {
public:
  CorrMatern32(const PointDimension d) : CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    double s = tinyNuggetOffDiag;
    double t;
    double prod = 1.0;
    for (PointDimension k = 0; k < d; ++k) {
      s += t = std::fabs(x1[k] - x2[k]) ;
      prod *= (1.0 + ExpoAlgo::unscale(t));
    }
    return prod*ExpoAlgo::expOfMinus(s);
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(Math::sqrt3);
  }
};
//-------------- Matern32Radial, draft not tested yet
template <typename ExpoAlgo>
class CorrMatern32Radial : public CorrelationFunction {
public:
  CorrMatern32Radial(const PointDimension d) : CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    const double s = norm2(x1,x2);
    return (1.0 + ExpoAlgo::unscale(s))*ExpoAlgo::expOfMinus(s);
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(Math::sqrt3);
  }
};

//-------------- Matern52
template <typename ExpoAlgo>
class CorrMatern52 : public CorrelationFunction {
  constexpr static double scaledOneOverThree = ExpoAlgo::unscale(ExpoAlgo::unscale(1.0L/3.0L));
public:
  CorrMatern52(const PointDimension d) : CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    double s = tinyNuggetOffDiag, t ;
    double prod = 1.0;
    for (PointDimension k = 0; k < d; ++k) {
      s += t = std::fabs(x1[k] - x2[k]);
      prod *= (1 + ExpoAlgo::unscale(t) + t*t*scaledOneOverThree);
      // with fma(x,y,z)=x*y+z, slower or identical, depending on compiler options
      //prod *= std::fma(std::fma(t, oneOverThree, 1.0), t, 1.0);
    }
    return prod * ExpoAlgo::expOfMinus(s);
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(Math::sqrt5);
  }
};
//-------------- Matern52Radial, draft not tested yet
template <typename ExpoAlgo>
class CorrMatern52Radial : public CorrelationFunction {
  constexpr static double scaledOneOverThree = ExpoAlgo::unscale(ExpoAlgo::unscale(1.0L/3.0L));
public:
  CorrMatern52Radial(const PointDimension d) : CorrelationFunction(d)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    const double s = norm2(x1,x2);
    return (1 + ExpoAlgo::unscale(s) + s*s*scaledOneOverThree)*ExpoAlgo::expOfMinus(s);
  }

  virtual Double scaling_factor() const override {
    return ExpoAlgo::scale(Math::sqrt5);
  }
};

//-------------- Powerexp
class CorrPowerexp : public CorrelationFunction {
public:
  const arma::vec& param;

  CorrPowerexp(const PointDimension d, const arma::vec& param) :
    CorrelationFunction(d), param(param)  {
  }

  virtual double corr(const Point& x1, const Point& x2) const noexcept override {
    double s = 0.0;
    for (PointDimension k = 0; k < d; ++k) s += std::pow(std::fabs(x1[k] - x2[k]), param[k+d]);
    return std::exp(-s);
  }

  virtual Double scaling_factor() const override {
    return 1.0L;
  }
};

//=========================================== CovarianceParameters
// class containing covariance parameters
// this class also do precomputations in order to fasten further covariance calculations

class CovarianceParameters {
public:
  using ScalingFactors=std::valarray<Double>;

private:
  const PointDimension d;
  const arma::vec param; //no &, copy to make kernel independent

  ScalingFactors createScalingFactors() const {
    ScalingFactors factors(d);
    const Double scalingCorr = corrFunction->scaling_factor();
    for(PointDimension k=0;k<d;++k) factors[k] = scalingCorr/param(k);
    return factors;
  }

  CorrelationFunction* getCorrelationFunction(std::string& covType) const {
         if (covType.compare("gauss.approx") == 0) {return new CorrGauss<ExpoApprox<5> >(d);}
    else if (covType.compare("gauss")==0) {return new CorrGauss<ExpoBase2>(d);}
    else if (covType.compare("exp.approx") == 0) {return new CorrExp<ExpoApprox<5> >(d);}
    else if (covType.compare("exp")==0) {return new CorrExp<ExpoBase2>(d);}
    else if (covType.compare("matern3_2.approx") == 0) {return new CorrMatern32<ExpoApprox<12> >(d);}
    else if (covType.compare("matern3_2") == 0) {return new CorrMatern32<ExpoBase2>(d);}
    else if (covType.compare("matern5_2.approx") == 0) {return new CorrMatern52<ExpoApprox<18> >(d);}
    else if (covType.compare("matern5_2") == 0) {return new CorrMatern52<ExpoBase2>(d);}
    else if (covType.compare("exp.radial") == 0) {return new CorrExpRadial<ExpoBase2>(d);}
    else if (covType.compare("matern3_2.radial") == 0) {return new CorrMatern32Radial<ExpoBase2>(d);}
    else if (covType.compare("matern5_2.radial") == 0) {return new CorrMatern52Radial<ExpoBase2>(d);}
    else if (covType.compare("powexp") == 0) {return new CorrPowerexp(d, param);}
    else if (covType.compare("white_noise") == 0) {return new CorrWhiteNoise(d);}
    else if (covType.compare("rational2") == 0) {return new CorrRational2(d);}
    else if (covType.compare("rational1") == 0) {return new CorrRational1(d);}
    //
    // --- temporary, tests of new kernels
    else if (covType.compare("rational2.new") == 0) {return new CorrGauss<ExpoApprox<0> >(d);}
    else if (covType.compare("rational1.new") == 0) {return new CorrExp<ExpoApprox<0> >(d);}
    else if (covType.compare("rational2square") == 0) {return new CorrGauss<ExpoApprox<1> >(d);}
    else if (covType.compare("rational1square") == 0) {return new CorrExp<ExpoApprox<1> >(d);}
    else if (covType.compare("rational2four") == 0) {return new CorrGauss<ExpoApprox<2> >(d);}
    else if (covType.compare("rational1four") == 0) {return new CorrExp<ExpoApprox<2> >(d);}
    //
    // --- temporary, legacy kernels for benchmarks and compatibility
    else if (covType.compare("gauss.legacy")==0) {return new CorrGauss<Expo>(d);}
    else if (covType.compare("exp.legacy")==0) {return new CorrExp<Expo>(d);}
    else if (covType.compare("matern3_2.legacy") == 0) {return new CorrMatern32<Expo>(d);}
    else if (covType.compare("matern5_2.legacy") == 0) {return new CorrMatern52<Expo>(d);}
    else if (covType.compare("exp.radial.legacy") == 0) {return new CorrExpRadial<Expo>(d);}
    else if (covType.compare("matern3_2.radial.legacy") == 0) {return new CorrMatern32Radial<Expo>(d);}
    else if (covType.compare("matern5_2.radial.legacy") == 0) {return new CorrMatern52Radial<Expo>(d);}
    else if (covType.compare("approx.exp") == 0) {return new CorrExp<ExpoApprox<4> >(d);}
    else if (covType.compare("approx.gauss") == 0) {return new CorrGauss<ExpoApprox<4> >(d);}
    else {
      //screen.warning("covType wrongly written, using exponential kernel");
      return new CorrExp<ExpoBase2>(d);}
  }

public:
  const double variance;
  const double inverseVariance;

  const CorrelationFunction* corrFunction;
  const ScalingFactors scalingFactors;

  CovarianceParameters(const PointDimension d, const arma::vec& param, const double variance, std::string covType) :
    d(d), param(param), variance(variance), inverseVariance(1/(variance+1e-100)),
    corrFunction(getCorrelationFunction(covType)),
    scalingFactors(createScalingFactors()) {
  }

  CovarianceParameters() = delete;

  ~CovarianceParameters() {
    delete corrFunction;
  }
  //-------------- this object is not copied nor moved
  CovarianceParameters (const CovarianceParameters &) = delete;
  CovarianceParameters& operator= (const CovarianceParameters &) = delete;

  CovarianceParameters (CovarianceParameters &&) = delete;
  CovarianceParameters& operator= (CovarianceParameters &&) = delete;

};
//======================================================== Points

class Points {
public:
  using Writable_Point_type = WritablePoint;
  using ReadOnly_Point_type = const Point;

private:
  PointsStorage _data{};
  PointDimension _d = 0;

  void fillWith(const arma::mat& source, const CovarianceParameters& covParam, const arma::rowvec& origin) {
    const Long nrows= source.n_rows;
    _d= source.n_cols;
    reserve(nrows, d);
    const CovarianceParameters::ScalingFactors& scalingFactors = covParam.scalingFactors;
    for(Long obs=0;obs<nrows;++obs) {
      for(PointDimension k=0;k<_d;++k)
        cell(obs,k) = (source.at(obs,k)-origin.at(k))*scalingFactors[k];
    }
  }

public:
  const PointDimension& d=_d;
  Points(const arma::mat& source, const CovarianceParameters& covParam, const arma::rowvec origin) {
    fillWith(source, covParam, origin);
  }
  Points(const arma::mat& source, const CovarianceParameters& covParam) {
    fillWith(source, covParam, arma::zeros<arma::rowvec>(source.n_cols));
  }

  Points() { } //used in splitter to obtain std::vector<Points>

#if (MATRIX_STORAGE==1)

  inline const ReadOnly_Point_type operator[](const std::size_t index) const {
    return _data.row(index);
    }
  inline Writable_Point_type operator[](const std::size_t index) {
    return _data.row(index);
  }
  inline std::size_t size() const {return _data.n_rows;}

  inline void reserve(const std::size_t rows, const std::size_t cols) {
    _data.set_size(rows, cols); _d= cols; }

  inline double& cell(const std::size_t row, const std::size_t col) { return _data.row(row)[col]; }

#else

  inline const ReadOnly_Point_type& operator[](const std::size_t index) const {
    return _data[index];
  }
  inline Writable_Point_type& operator[](const std::size_t index) {
    return _data[index];
  }
  inline std::size_t size() const {return _data.size();}
  inline void resize(const std::size_t length) { _data.resize(length); }

  inline void reserve(const std::size_t rows, const std::size_t cols) {
    _data.resize(rows);
    _d= cols;
    for(Long i=0; i<rows; ++i) _data[i].resize(cols);
   }

  inline double& cell(const std::size_t row, const std::size_t col) { return _data[row][col]; }
#endif

  Points (const Points &other) = default;
  Points& operator= (const Points &other) = default;

  Points (Points &&other) = default;
  Points& operator= (Points &&other) = default;
};

//============================================================  Covariance

class Covariance {
  const CovarianceParameters& params;
  const CorrelationFunction* corrFunction;

public:
  using NuggetVector = arma::vec;
  constexpr static double diagonalValue = 1.0 + tinyNuggetOnDiag;

  Covariance(const CovarianceParameters& params) : params(params), corrFunction(params.corrFunction) {}

  void fillAllocatedDiagonal(arma::mat& matrixToFill, const NuggetVector& nugget) const noexcept {
    const Long n = matrixToFill.n_rows, nuggetSize=nugget.size();
    if (nuggetSize==0) {
      for (Long i = 0; i < n; ++i) matrixToFill.at(i,i) = diagonalValue;
      }
    else if (nuggetSize==1) {
      double diagvalueNugget = diagonalValue + nugget[0]*params.inverseVariance;
      for (Long i = 0; i < n; ++i) matrixToFill.at(i,i) = diagvalueNugget;
      }
    else if (nuggetSize==n) {
      for (Long i = 0; i < n; ++i) matrixToFill.at(i,i) = diagonalValue + nugget[i]*params.inverseVariance;
      }
    else {
      for (Long i = 0; i < n; ++i) matrixToFill.at(i,i) = diagonalValue + nugget[i%nuggetSize]*params.inverseVariance;
    }
  }

  void fillAllocatedCorrMatrix(arma::mat& matrixToFill, const Points& points, const NuggetVector& nugget) const noexcept {
    // noexcept, assume that matrixToFill is a correctly allocated square matrix of size points.size()
    fillAllocatedDiagonal(matrixToFill, nugget);
    for (Long i = 0; i < points.size(); ++i) // parallelizable
      for (Long j = 0; j < i; ++j)
        matrixToFill.at(i,j) = matrixToFill.at(j,i) = corrFunction->corr(points[i], points[j]);
  }
  void fillAllocatedCrossCorrelations(arma::mat& matrixToFill, const Points& pointsA, const Points& pointsB) const noexcept {
    // noexcept, assume that matrixToFill is a correctly allocated matrix of size pointsA.size() x pointsB.size()
    // Warning: part of critical importance for the performance of the Algo
    for (Long j = 0; j < pointsB.size(); ++j) // arma::mat is column major ordering
      for (Long i = 0; i < pointsA.size(); ++i)
        matrixToFill.at(i,j) = corrFunction->corr(pointsA[i], pointsB[j]);
  }


  void fillCorrMatrix(arma::mat& matrixToFill, const Points& points, const NuggetVector& nugget) const {
    try{
      matrixToFill.set_size(points.size(),points.size());
      fillAllocatedCorrMatrix(matrixToFill, points, nugget);
    }
    catch(const std::exception &e) {
      Screen::error("error in fillCorrMatrix", e);
      throw;
    }
  }

  void fillCrossCorrelations(arma::mat& matrixToFill, const Points& pointsA, const Points& pointsB) const {
    try{
    matrixToFill.set_size(pointsA.size(),pointsB.size());
    fillAllocatedCrossCorrelations(matrixToFill, pointsA, pointsB);
    }
    catch(const std::exception &e) {
      Screen::error("error in fillCrossCorrelations", e);
      throw;
    }
  }
};

// } //end namespace nestedKrig

//======================================== exports, outside namespace
//[[Rcpp::export]]
arma::mat getCorrMatrix(arma::mat X, arma::vec param, std::string covType) {
  using namespace nestedKrig;
  const long d= X.n_cols;
  const Covariance::NuggetVector emptyNugget{};
  const CovarianceParameters covParams(d, param, 1.0, covType);
  const Covariance kernel(covParams);
  arma::mat K;
  kernel.fillCorrMatrix(K, Points(X, covParams), emptyNugget);
  return K;
}

//[[Rcpp::export]]
arma::mat getCrossCorrMatrix(arma::mat X1, arma::mat X2, arma::vec param, std::string covType) {
  using namespace nestedKrig;
  const long d= X1.n_cols;
  const CovarianceParameters covParams(d, param, 1.0, covType);
  const Covariance kernel(covParams);
  arma::mat K;
  kernel.fillCrossCorrelations(K, Points(X1, covParams), Points(X2, covParams));
  return K;
}

#endif /* COVARIANCE_HPP */

/*  Comments if new storages are required:
 *  if MATRIX_STORAGE==0: PointsStorage can be any of std::vetor<Type>
 *     then it needs read/write[], copy ctor, default ctor, resize(), size() to get the number of points
 *  if MATRIX_STORAGE==1: PointsStorage can be any of arma::mat type
 *     then it needs read/write row(), set_size(), n_cols, ..
 *  Point needs read only [] so that vector, arma::vec, double*, arma::subview_row<double> are acceptable
 *  Point is only used in read-only operations
 *  WritablePoints needs [], copy and = and resize in the case of vector storage
 */

