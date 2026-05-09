#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "MLPKriging_binding.hpp"
#include "Optim_binding.hpp"
#include "Params_binding.hpp"
#include "WarpKriging_binding.hpp"
#include "libKriging/KrigingLoader.hpp"
#include "mex.h"  // cf https://fr.mathworks.com/help/
#include "tools/MxException.hpp"
#include "tools/MxMapper.hpp"
#include "tools/string_hash.hpp"

void help_page() {
  mexPrintf(R"(
mLibKriging help page
  the Octave/Matlab interface for libKriging

  see https://github.com/libKriging/libKriging
  
)");
}

/*
 clear all
 y = randn (20,1);
 X = randn (20,1);
 X2 = randn (20,1);
 a=LinearRegression();
 a.fit(y,X);
 [y2,stderr] = a.predict(X2);

 load "y.mat"
 load "X.mat"
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) try {
  const char* nm = mexFunctionName();
#ifdef MEX_DEBUG
  mexPrintf("  You called MEX function: %s\n", nm);
  mexPrintf("  with %d inputs and %d outputs\n", nrhs, nlhs);
  // mexCallMATLAB(0, NULL, 1, (mxArray**)&prhs[0], "disp");
#endif
  if (std::strcmp(nm, "mLibKriging") != 0) {
    throw MxException(LOCATION(), "mLibKriging:badMexFunction", "you call mLibKriging with an illegal name");
  }

  if (nrhs < 1)
    return help_page();

  MxMapper input{"Input", std::min(1, nrhs), const_cast<mxArray**>(prhs)};
  std::string command = input.get<std::string>(0, "command");

#ifdef MEX_DEBUG
  mexPrintf("  with command: %s\n", command.c_str());
#endif

  switch (fnv_hash(command)) {
    case "help"_hash:
      return help_page();

    case "class_saved"_hash: {
      MxMapper input{"Input", nrhs - 1, const_cast<mxArray**>(prhs + 1), RequiresArg::Exactly{1}};
      MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
      (void)output;
      const auto filename = input.get<std::string>(0, "filename");
      std::string klass;
      switch (KrigingLoader::describe(filename)) {
        case KrigingLoader::KrigingType::Kriging:
        case KrigingLoader::KrigingType::NuggetKriging:
        case KrigingLoader::KrigingType::NoiseKriging:
          klass = "Kriging";
          break;
        case KrigingLoader::KrigingType::WarpKriging:
          klass = "WarpKriging";
          break;
        case KrigingLoader::KrigingType::MLPKriging:
          klass = "MLPKriging";
          break;
        case KrigingLoader::KrigingType::Unknown:
          mexErrMsgIdAndTxt("mLibKriging:class_saved", "Unknown Kriging type in file");
      }
      plhs[0] = mxCreateString(klass.c_str());
      return;
    }

    case "Params::new"_hash:
      return ParamsBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Params::delete"_hash:
      return ParamsBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Params::display"_hash:
      return ParamsBinding::display(nlhs, plhs, nrhs - 1, prhs + 1);

    case "LinearRegression::new"_hash:
      return LinearRegressionBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::delete"_hash:
      return LinearRegressionBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::fit"_hash:
      return LinearRegressionBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::predict"_hash:
      return LinearRegressionBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);

    case "Kriging::new"_hash:
      return KrigingBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::copy"_hash:
      return KrigingBinding::copy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::delete"_hash:
      return KrigingBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::fit"_hash:
      return KrigingBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::predict"_hash:
      return KrigingBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::simulate"_hash:
      return KrigingBinding::simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::update"_hash:
      return KrigingBinding::update(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::update_simulate"_hash:
      return KrigingBinding::update_simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::summary"_hash:
      return KrigingBinding::summary(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::save"_hash:
      return KrigingBinding::save(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::load"_hash:
      return KrigingBinding::load(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::leaveOneOutFun"_hash:
      return KrigingBinding::leaveOneOutFun(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::leaveOneOutVec"_hash:
      return KrigingBinding::leaveOneOutVec(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logLikelihoodFun"_hash:
      return KrigingBinding::logLikelihoodFun(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logMargPostFun"_hash:
      return KrigingBinding::logMargPostFun(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::leaveOneOut"_hash:
      return KrigingBinding::leaveOneOut(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logLikelihood"_hash:
      return KrigingBinding::logLikelihood(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logMargPost"_hash:
      return KrigingBinding::logMargPost(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::covMat"_hash:
      return KrigingBinding::covMat(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::model"_hash:
      return KrigingBinding::model(nlhs, plhs, nrhs - 1, prhs + 1);

    case "Kriging::kernel"_hash:
      return KrigingBinding::kernel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::optim"_hash:
      return KrigingBinding::optim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::objective"_hash:
      return KrigingBinding::objective(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::X"_hash:
      return KrigingBinding::X(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::centerX"_hash:
      return KrigingBinding::centerX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::scaleX"_hash:
      return KrigingBinding::scaleX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::y"_hash:
      return KrigingBinding::y(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::centerY"_hash:
      return KrigingBinding::centerY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::scaleY"_hash:
      return KrigingBinding::scaleY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::normalize"_hash:
      return KrigingBinding::normalize(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::regmodel"_hash:
      return KrigingBinding::regmodel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::F"_hash:
      return KrigingBinding::F(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::T"_hash:
      return KrigingBinding::T(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::M"_hash:
      return KrigingBinding::M(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::z"_hash:
      return KrigingBinding::z(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::beta"_hash:
      return KrigingBinding::beta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_beta_estim"_hash:
      return KrigingBinding::is_beta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::theta"_hash:
      return KrigingBinding::theta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_theta_estim"_hash:
      return KrigingBinding::is_theta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::sigma2"_hash:
      return KrigingBinding::sigma2(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_sigma2_estim "_hash:
      return KrigingBinding::is_sigma2_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::noise_model"_hash:
      return KrigingBinding::noise_model(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::nugget"_hash:
      return KrigingBinding::nugget(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_nugget_estim"_hash:
      return KrigingBinding::is_nugget_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::noise"_hash:
      return KrigingBinding::noise(nlhs, plhs, nrhs - 1, prhs + 1);

    case "WarpKriging::new"_hash:
      return WarpKrigingBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::copy"_hash:
      return WarpKrigingBinding::copy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::delete"_hash:
      return WarpKrigingBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::fit"_hash:
      return WarpKrigingBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::predict"_hash:
      return WarpKrigingBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::simulate"_hash:
      return WarpKrigingBinding::simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::update_simulate"_hash:
      return WarpKrigingBinding::update_simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::update"_hash:
      return WarpKrigingBinding::update(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::summary"_hash:
      return WarpKrigingBinding::summary(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::logLikelihoodFun"_hash:
      return WarpKrigingBinding::logLikelihoodFun(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::logLikelihood"_hash:
      return WarpKrigingBinding::logLikelihood(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::kernel"_hash:
      return WarpKrigingBinding::kernel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::X"_hash:
      return WarpKrigingBinding::X(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::y"_hash:
      return WarpKrigingBinding::y(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::centerX"_hash:
      return WarpKrigingBinding::centerX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::scaleX"_hash:
      return WarpKrigingBinding::scaleX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::centerY"_hash:
      return WarpKrigingBinding::centerY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::scaleY"_hash:
      return WarpKrigingBinding::scaleY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::normalize"_hash:
      return WarpKrigingBinding::normalize(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::regmodel"_hash:
      return WarpKrigingBinding::regmodel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::F"_hash:
      return WarpKrigingBinding::F(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::T"_hash:
      return WarpKrigingBinding::T(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::M"_hash:
      return WarpKrigingBinding::M(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::z"_hash:
      return WarpKrigingBinding::z(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::beta"_hash:
      return WarpKrigingBinding::beta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::theta"_hash:
      return WarpKrigingBinding::theta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::sigma2"_hash:
      return WarpKrigingBinding::sigma2(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::is_fitted"_hash:
      return WarpKrigingBinding::is_fitted(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::feature_dim"_hash:
      return WarpKrigingBinding::feature_dim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::warping"_hash:
      return WarpKrigingBinding::warping(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::save"_hash:
      return WarpKrigingBinding::save(nlhs, plhs, nrhs - 1, prhs + 1);
    case "WarpKriging::load"_hash:
      return WarpKrigingBinding::load(nlhs, plhs, nrhs - 1, prhs + 1);

    case "MLPKriging::new"_hash:
      return MLPKrigingBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::delete"_hash:
      return MLPKrigingBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::fit"_hash:
      return MLPKrigingBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::predict"_hash:
      return MLPKrigingBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::simulate"_hash:
      return MLPKrigingBinding::simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::update_simulate"_hash:
      return MLPKrigingBinding::update_simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::update"_hash:
      return MLPKrigingBinding::update(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::summary"_hash:
      return MLPKrigingBinding::summary(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::logLikelihoodFun"_hash:
      return MLPKrigingBinding::logLikelihoodFun(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::logLikelihood"_hash:
      return MLPKrigingBinding::logLikelihood(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::kernel"_hash:
      return MLPKrigingBinding::kernel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::X"_hash:
      return MLPKrigingBinding::X(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::y"_hash:
      return MLPKrigingBinding::y(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::centerX"_hash:
      return MLPKrigingBinding::centerX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::scaleX"_hash:
      return MLPKrigingBinding::scaleX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::centerY"_hash:
      return MLPKrigingBinding::centerY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::scaleY"_hash:
      return MLPKrigingBinding::scaleY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::normalize"_hash:
      return MLPKrigingBinding::normalize(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::regmodel"_hash:
      return MLPKrigingBinding::regmodel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::F"_hash:
      return MLPKrigingBinding::F(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::T"_hash:
      return MLPKrigingBinding::T(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::M"_hash:
      return MLPKrigingBinding::M(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::z"_hash:
      return MLPKrigingBinding::z(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::beta"_hash:
      return MLPKrigingBinding::beta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::theta"_hash:
      return MLPKrigingBinding::theta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::sigma2"_hash:
      return MLPKrigingBinding::sigma2(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::is_fitted"_hash:
      return MLPKrigingBinding::is_fitted(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::feature_dim"_hash:
      return MLPKrigingBinding::feature_dim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::hidden_dims"_hash:
      return MLPKrigingBinding::hidden_dims(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::activation"_hash:
      return MLPKrigingBinding::activation(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::copy"_hash:
      return MLPKrigingBinding::copy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::save"_hash:
      return MLPKrigingBinding::save(nlhs, plhs, nrhs - 1, prhs + 1);
    case "MLPKriging::load"_hash:
      return MLPKrigingBinding::load(nlhs, plhs, nrhs - 1, prhs + 1);

    case "Optim::is_reparametrized"_hash:
      return OptimBinding::is_reparametrized(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::use_reparametrize"_hash:
      return OptimBinding::use_reparametrize(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_theta_lower_factor"_hash:
      return OptimBinding::get_theta_lower_factor(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_theta_lower_factor"_hash:
      return OptimBinding::set_theta_lower_factor(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_theta_upper_factor"_hash:
      return OptimBinding::get_theta_upper_factor(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_theta_upper_factor"_hash:
      return OptimBinding::set_theta_upper_factor(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::variogram_bounds_heuristic_used"_hash:
      return OptimBinding::variogram_bounds_heuristic_used(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::use_variogram_bounds_heuristic"_hash:
      return OptimBinding::use_variogram_bounds_heuristic(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_log_level"_hash:
      return OptimBinding::get_log_level(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_log_level"_hash:
      return OptimBinding::set_log_level(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_max_iteration"_hash:
      return OptimBinding::get_max_iteration(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_max_iteration"_hash:
      return OptimBinding::set_max_iteration(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_gradient_tolerance"_hash:
      return OptimBinding::get_gradient_tolerance(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_gradient_tolerance"_hash:
      return OptimBinding::set_gradient_tolerance(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_objective_rel_tolerance"_hash:
      return OptimBinding::get_objective_rel_tolerance(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_objective_rel_tolerance"_hash:
      return OptimBinding::set_objective_rel_tolerance(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_thread_start_delay_ms"_hash:
      return OptimBinding::get_thread_start_delay_ms(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_thread_start_delay_ms"_hash:
      return OptimBinding::set_thread_start_delay_ms(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::get_thread_pool_size"_hash:
      return OptimBinding::get_thread_pool_size(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Optim::set_thread_pool_size"_hash:
      return OptimBinding::set_thread_pool_size(nlhs, plhs, nrhs - 1, prhs + 1);

    default:
      throw MxException(LOCATION(), "mLibKriging:noRoute", "No route to such command [", command, "]");
  }
} catch (MxException& e) {
  mexErrMsgIdAndTxt(e.id, e.msg.c_str());
} catch (std::exception& e) {
  mexErrMsgIdAndTxt("mLibKriging:kernelException", e.what());
} catch (...) {  // catch everything even end-of-scope event
  mexErrMsgIdAndTxt("mLibKriging:exception", "unexpected exception");
}
