#include "NestedKriging_binding.hpp"

#include "libKriging/NestedKriging.hpp"
#include "libKriging/Trend.hpp"

#include "Params.hpp"
#include "common_binding.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

static Kriging::Parameters makeParameters(std::optional<Params*> dict) {
  if (dict) {
    const Params& params = *dict.value();
    return Kriging::Parameters{params.get<double>("sigma2"),
                               params.get<bool>("is_sigma2_estim").value_or(true),
                               params.get<arma::mat>("theta"),
                               params.get<bool>("is_theta_estim").value_or(true),
                               params.get<arma::mat>("beta"),
                               params.get<bool>("is_beta_estim").value_or(true),
                               params.get<double>("nugget"),
                               params.get<bool>("is_nugget_estim").value_or(true)};
  }
  return Kriging::Parameters{};
}

// Convert a MATLAB cell array of strings to std::vector<std::string>
static std::vector<std::string> cellToStringVec(const mxArray* cell, const char* param_name) {
  if (!mxIsCell(cell))
    throw std::runtime_error(std::string(param_name) + " must be a cell array of strings");
  const size_t n = mxGetNumberOfElements(cell);
  std::vector<std::string> result(n);
  for (size_t i = 0; i < n; ++i) {
    const mxArray* item = mxGetCell(cell, static_cast<mwIndex>(i));
    char* str = mxArrayToString(item);
    if (!str)
      throw std::runtime_error(std::string(param_name) + " contains a non-string element");
    result[i] = str;
    mxFree(str);
  }
  return result;
}

static NestedKriging::Partition parsePartition(const std::string& s) {
  if (s == "kmeans")
    return NestedKriging::Partition::KMeans;
  if (s == "random")
    return NestedKriging::Partition::Random;
  throw std::runtime_error("Unknown partition: '" + s + "'. Expected 'kmeans' or 'random'.");
}

namespace NestedKrigingBinding {

// NestedKriging::new(y, X, kernel, nb_groups, [aggregation], [partition], [seed],
//                    [regmodel], [optim], [objective], [parameters], [warping_cell])
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{4, 12}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};

  auto y = input.get<arma::vec>(0, "vector");
  auto X = input.get<arma::mat>(1, "matrix");
  auto kernel = input.get<std::string>(2, "kernel");
  const auto nb_groups = static_cast<arma::uword>(input.get<double>(3, "nb_groups"));
  const auto aggregation
      = NestedKriging::aggregationFromString(input.getOptional<std::string>(4, "aggregation").value_or("NK"));
  const auto partition = parsePartition(input.getOptional<std::string>(5, "partition").value_or("kmeans"));
  const auto seed = static_cast<int>(input.getOptional<double>(6, "seed").value_or(123));
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(7, "regression model").value_or("constant"));
  const auto optim = input.getOptional<std::string>(8, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(9, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(10, "parameters"));
  std::vector<std::string> warping;
  if (nrhs > 11 && !mxIsEmpty(prhs[11]))
    warping = cellToStringVec(prhs[11], "warping");

  NestedKriging nk_obj(
      y, X, kernel, nb_groups, aggregation, partition, seed, regmodel, optim, objective, parameters, warping);
  auto nk = buildObject<NestedKriging>(std::move(nk_obj));
  output.set(0, nk, "new object reference");
}

void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  destroyObject(input.get<uint64_t>(0, "object reference"));
  output.set(0, EmptyObject{}, "deleted object reference");
}

// NestedKriging::fit(ref, y, X, nb_groups, [regmodel], [optim], [objective], [parameters], [warping_cell])
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{4, 9}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* nk = input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference");
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(4, "regression model").value_or("constant"));
  const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
  std::vector<std::string> warping;
  if (nrhs > 8 && !mxIsEmpty(prhs[8]))
    warping = cellToStringVec(prhs[8], "warping");
  nk->fit(input.get<arma::vec>(1, "vector"),
          input.get<arma::mat>(2, "matrix"),
          static_cast<arma::uword>(input.get<double>(3, "nb_groups")),
          regmodel,
          optim,
          objective,
          parameters,
          warping);
}

// [mean, [stdev]] = NestedKriging::predict(ref, X)
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* nk = input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference");
  const bool return_stdev = (nlhs >= 2);
  auto [mean, stdev] = nk->predict(input.get<arma::mat>(1, "matrix"), return_stdev);
  output.set(0, mean, "predicted response");
  output.setOptional(1, stdev, "stdev vector");
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* nk = input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference");
  output.set(0, nk->summary(), "summary string");
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->kernel(), "kernel");
}

void aggregation(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0,
             NestedKriging::aggregationToString(
                 input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->aggregation()),
             "aggregation");
}

void nb_groups(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0,
             static_cast<double>(input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->nb_groups()),
             "nb_groups");
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->theta(), "theta");
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->sigma2(), "sigma2");
}

void beta0(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, input.getObjectFromRef<NestedKriging>(0, "NestedKriging reference")->beta0(), "beta0");
}

}  // namespace NestedKrigingBinding
