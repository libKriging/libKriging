#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingLoader.hpp"
#include "libKriging/utils/hdf5utils.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>

KrigingLoader::KrigingType KrigingLoader::describe(std::string filename) {
  uint32_t version;
  loadFromHdf5(version, arma::hdf5_name(filename, "version"));
  if (version != 1) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 1"));
  }

  std::string content;
  loadFromHdf5(content, arma::hdf5_name(filename, "content"));
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