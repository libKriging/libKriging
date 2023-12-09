#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingLoader.hpp"
#include "libKriging/utils/hdf5utils.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>

KrigingLoader::KrigingType KrigingLoader::describe(std::string filename) {
  const bool use_hdf5 = false;
  std::string content;

  if (use_hdf5) {
    uint32_t version;
    loadFromHdf5(version, arma::hdf5_name(filename, "version"));
    if (version != 1) {
      throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 1"));
    }
    loadFromHdf5(content, arma::hdf5_name(filename, "content"));
  } else {
    std::ifstream f(filename);
    nlohmann::json j = nlohmann::json::parse(f);

    uint32_t version = j["version"].template get<uint32_t>();
    if (version != 2) {
      throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
    }
    content = j["content"].template get<std::string>();
  }

  if (content == "Kriging") {
    return KrigingType::Kriging;
  } else if (content == "NoiseKriging") {
    return KrigingType::NoiseKriging;
  } else if (content == "NuggetKriging") {
    return KrigingType::NuggetKriging;
  } else {
    return KrigingType::Unknown;
  }
}