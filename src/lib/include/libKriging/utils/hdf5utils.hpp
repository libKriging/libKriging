#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include <armadillo>

void saveToHdf5(const std::string& s, const arma::hdf5_name& location);

void saveToHdf5(const bool& t, const arma::hdf5_name& location);

void saveToHdf5(const uint32_t& t, const arma::hdf5_name& location);

void saveToHdf5(const double& t, const arma::hdf5_name& location);

void loadFromHdf5(std::string& s, const arma::hdf5_name& location);

void loadFromHdf5(bool& t, const arma::hdf5_name& location);

void loadFromHdf5(uint32_t& t, const arma::hdf5_name& location);

void loadFromHdf5(double& t, const arma::hdf5_name& location);

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP