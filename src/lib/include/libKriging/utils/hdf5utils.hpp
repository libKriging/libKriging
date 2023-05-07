#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include <armadillo>

void saveToHdf5(const std::string &s, const arma::hdf5_name &location) {
    const auto size = s.size();
    arma::Col<uint8_t> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        v[i] = s[i];
    }
    if (!v.save(location))
        throw std::runtime_error("Cannot save " + location.dsname + " in " + location.filename);
}

void saveToHdf5(const bool &t, const arma::hdf5_name &location) {
    arma::Col<uint8_t> v(1);
    v[0] = static_cast<uint8_t>(t);
    if (!v.save(location))
        throw std::runtime_error("Cannot save " + location.dsname + " in " + location.filename);
}

void saveToHdf5(const uint32_t &t, const arma::hdf5_name &location) {
    arma::Col<uint32_t> v(1);
    v[0] = t;
    if (!v.save(location))
        throw std::runtime_error("Cannot save " + location.dsname + " in " + location.filename);
}

void saveToHdf5(const double &t, const arma::hdf5_name &location) {
    arma::Col<double> v(1);
    v[0] = t;
    if (!v.save(location))
        throw std::runtime_error("Cannot save " + location.dsname + " in " + location.filename);
}

void loadFromHdf5(std::string &s, const arma::hdf5_name &location) {
    arma::Col<uint8_t> v;
    bool load_okay = v.load(location);
    if (load_okay) {
        s.resize(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            s[i] = v[i];
        }
    } else {
        throw std::runtime_error("Cannot load " + location.dsname + " in " + location.filename);
    }
}

void loadFromHdf5(bool &t, const arma::hdf5_name &location) {
    arma::Col<uint8_t> v;
    bool load_okay = v.load(location);
    if (load_okay) {
        t = static_cast<bool>(v[0]);
    } else {
        throw std::runtime_error("Cannot load " + location.dsname + " in " + location.filename);
    }
}

void loadFromHdf5(uint32_t &t, const arma::hdf5_name &location) {
    arma::Col<uint32_t> v;
    bool load_okay = v.load(location);
    if (load_okay) {
        t = v[0];
    } else {
        throw std::runtime_error("Cannot load " + location.dsname + " in " + location.filename);
    }
}

void loadFromHdf5(double &t, const arma::hdf5_name &location) {
    arma::Col<double> v;
    bool load_okay = v.load(location);
    if (load_okay) {
        t = v[0];
    } else {
        throw std::runtime_error("Cannot load " + location.dsname + " in " + location.filename);
    }
}

#endif // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HDF5UTILS_HPP