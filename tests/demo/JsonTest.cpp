// Copyright (c) 2018-2023 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/json/

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tao/json.hpp>
#include <tao/json/value.hpp>
#include "tao/json/events/from_file.hpp"
#include "tao/json/events/from_value.hpp"
#include "tao/json/events/to_value.hpp"
#include "tao/json/events/validate_event_order.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/utils/jsonutils.hpp"


struct my_type {
    std::vector<double> values;
};

tao::json::value to_value(const my_type &t) {
    tao::binary_view vvv = tao::to_binary_view(reinterpret_cast<const std::byte *>(t.values.data()),
                                               sizeof(double) * t.values.size());
    auto x = tao::json::internal::base64(vvv);
    return
            {
                    {"size",   t.values.size()},
                    {"values", x}
            };
}

int main(int argc, char **argv) {
    const int n = 2;
    arma::rowvec a(n, arma::fill::randn);

    const tao::json::value v3 = {
            {"version",    2},
            {"content",    "Kriging"},
            {"covType",    "matern3_2"},
            {"X",          to_value(a)},
            {"centerX",    to_value(a)},
            {"m_est_beta", false},
            {"est_beta",   to_value(a)},
//            {
//             "sub-object", {
//                                   {"value", 1},
//                                   {"frobnicate", true}
//                           }
//            }
    };
    tao::json::to_stream(std::cout, v3);

    return 0;
}