#include "libKriging/utils/jsonutils.hpp"

#include <cstddef>

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

#include <iostream>

bool isLittleEndian() {
    uint16_t number = 0x01;
    uint8_t *bytePtr = reinterpret_cast<uint8_t *>(&number);
    return (*bytePtr == 0x01);
}


template<typename T>
std::vector<uint8_t> serialize(const std::vector<T> &data) {
    std::vector<uint8_t> result;
    result.reserve(data.size() * sizeof(T));

    for (const auto &value: data) {
        // Use memcpy to copy the raw bytes of the value into the result vector
        uint8_t bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));

        // If the endianness is big, reverse the byte order
        if (!isLittleEndian()) {
            std::reverse(std::begin(bytes), std::end(bytes));
        }

        // Append the bytes to the result vector
        result.insert(result.end(), std::begin(bytes), std::end(bytes));
    }

    return result;
}

template<typename T>
std::vector<T> deserialize(const std::vector<uint8_t> &data) {
    std::vector<T> result;
    result.reserve(data.size() / sizeof(T));

    for (size_t i = 0; i < data.size(); i += sizeof(T)) {
        // Copy the raw bytes from the data vector into a temporary array
        uint8_t bytes[sizeof(T)];
        std::copy(data.begin() + i, data.begin() + i + sizeof(T), std::begin(bytes));

        // If the endianness is big, reverse the byte order
        if (!isLittleEndian()) {
            std::reverse(std::begin(bytes), std::end(bytes));
        }

        // Use memcpy to convert the raw bytes back to the original type and push it to the result vector
        T value;
        std::memcpy(&value, bytes, sizeof(T));
        result.push_back(value);
    }

    return result;
}


tao::json::value to_value(const arma::rowvec &t) {
    tao::binary_view vvv = tao::to_binary_view(reinterpret_cast<const std::byte *>(t.memptr()),
                                               sizeof(double) * t.size());
    auto x = tao::json::internal::base64(vvv);
    return
            {
                    {"size",   t.size()},
                    {"values", x}
            };
}
