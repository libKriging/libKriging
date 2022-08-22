#include "Params_binding.hpp"
#include "Params.hpp"

#include "tools/MxException.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

namespace ParamsBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::KVPairs{}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto ref = buildObject<Params>();
  auto* params = ObjectCollector::getObject<Params>(ref);
  for (int i = 0; i < nrhs; i += 2) {
    std::string key = input.get<std::string>(i, "key");
    eMxType value_type = input.getType(i + 1, "value");
    switch (value_type) {
      case eMxType::String:
        params->set(key, input.get<std::string>(i + 1, "string value"));
        break;
      case eMxType::Matrix:
        params->set(key, input.get<arma::mat>(i + 1, "matrix value"));
        break;
      case eMxType::Uint64:
        params->set(key, input.get<uint64_t>(i + 1, "uint64 value"));
        break;
      case eMxType::Int32:
        params->set(key, input.get<int32_t>(i + 1, "int32 value"));
        break;
      case eMxType::Logical:
        params->set(key, input.get<bool>(i + 1, "logical value"));
        break;
      case eMxType::Scalar:
        params->set(key, input.get<double>(i + 1, "double value"));
        break;
      default:
        throw MxException(LOCATION(), "mLibKriging:Params", "Unsupported type value type '", value_type, "'");
    }
  }
  output.set(0, ref, "new object reference");
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

void display(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* params = input.getObject<Params>(0, "Params reference");
  params->display();
}
}  // namespace ParamsBinding
