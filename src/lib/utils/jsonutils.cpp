#include "libKriging/utils/jsonutils.hpp"

#include <cstddef>

#include <algorithm>
#include <cstring>
#include <vector>

#include "libKriging/utils/nlohmann/json.hpp"

#include "base64.h"

bool isLittleEndian() {
  uint16_t number = 0x01;
  uint8_t* bytePtr = reinterpret_cast<uint8_t*>(&number);
  return (*bytePtr == 0x01);
}

template <typename T>
std::vector<uint8_t> serialize(const T* data, const std::size_t size) {
  std::vector<uint8_t> result;
  result.reserve(size * sizeof(T));

  for (std::size_t i = 0; i < size; ++i) {
    const T value = data[i];
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

template <typename T>
std::vector<T> deserialize(const uint8_t* data, std::size_t size) {
  std::vector<T> result;
  result.reserve(size / sizeof(T));

  for (size_t i = 0; i < size; i += sizeof(T)) {
    // Copy the raw bytes from the data vector into a temporary array
    uint8_t bytes[sizeof(T)];
    std::copy(data + i, data + i + sizeof(T), std::begin(bytes));

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

nlohmann::json to_json(const arma::rowvec& t) {
  auto data = serialize(t.memptr(), t.size());
  std::string base64_data = base64_encode(data.data(), data.size(), false);
  return {{"type", "rowvec"}, {"size", t.size()}, {"base64_data", base64_data}};
}

nlohmann::json to_json(const arma::colvec& t) {
  auto data = serialize(t.memptr(), t.size());
  std::string base64_data = base64_encode(data.data(), data.size(), false);
  return {{"type", "colvec"}, {"size", t.size()}, {"base64_data", base64_data}};
}

nlohmann::json to_json(const arma::mat& t) {
  auto data = serialize(t.memptr(), t.size());
  std::string base64_data = base64_encode(data.data(), data.size(), false);
  return {{"type", "mat"}, {"n_rows", t.n_rows}, {"n_cols", t.n_cols}, {"base64_data", base64_data}};
}

arma::rowvec rowvec_from_json(const nlohmann::json& json_node) {
  assert(json_node["type"] == "rowvec");
  std::size_t size = json_node["size"];
  std::string base64_data = json_node["base64_data"];
  const std::string data = base64_decode(base64_data);
  std::vector<double> raw_data = deserialize<double>(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  assert(raw_data.size() == size);
  return {raw_data};
}

arma::colvec colvec_from_json(const nlohmann::json& json_node) {
  assert(json_node["type"] == "colvec");
  std::size_t size = json_node["size"];
  std::string base64_data = json_node["base64_data"];
  const std::string data = base64_decode(base64_data);
  std::vector<double> raw_data = deserialize<double>(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  assert(raw_data.size() == size);
  return {raw_data};
}

arma::mat mat_from_json(const nlohmann::json& json_node) {
  assert(json_node["type"] == "mat");
  arma::uword n_rows = json_node["n_rows"];
  arma::uword n_cols = json_node["n_cols"];
  std::string base64_data = json_node["base64_data"];
  const std::string data = base64_decode(base64_data);
  std::vector<double> raw_data = deserialize<double>(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  assert(raw_data.size() == n_cols * n_rows);
  return {raw_data.data(), n_rows, n_cols};
}
