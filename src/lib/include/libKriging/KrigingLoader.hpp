//
// Created by Pascal Havé on 2023-05-15.
//

#ifndef LIBKRIGING_KRIGINGLOADER_HPP
#define LIBKRIGING_KRIGINGLOADER_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

struct KrigingLoader {
  enum class LIBKRIGING_EXPORT KrigingType { Kriging, NoiseKriging, NuggetKriging, WarpKriging, MLPKriging, Unknown };

  LIBKRIGING_EXPORT static KrigingType describe(std::string filename);
};

#endif  // LIBKRIGING_KRIGINGLOADER_HPP
