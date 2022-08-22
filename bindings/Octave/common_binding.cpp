#include "common_binding.hpp"

bool flag_output_compliance(MxMapper& input, int I, const char* msg, const MxMapper& output, int output_position) {
  const auto flag = input.template getOptional<bool>(I, msg);
  if (flag) {
    const bool flag_value = flag.value();
    if (flag_value) {
      if (output.count() <= output_position) {
        throw MxException(LOCATION(),
                          "mLibKriging:inconsistentOutput",
                          MxMapper::parameterStr(I, msg),
                          " is set without related output");
      }
    }
    return flag_value;
  } else {
    return output.count() > output_position;
  }
}
