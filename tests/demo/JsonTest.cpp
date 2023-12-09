#include <iostream>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

int main(int argc, char** argv) {
  const int n = 2000;
  arma::rowvec a(n, arma::fill::randn);

  nlohmann::json j;

  // add a number that is stored as double (note the implicit conversion of j to an object)
  j["pi"] = 3.141;

  // add a Boolean that is stored as bool
  j["happy"] = to_json(a);

  // add a string that is stored as std::string
  j["name"] = "Niels";

  // add another null object by passing nullptr
  j["nothing"] = nullptr;

  // add an object inside the object
  j["answer"]["everything"] = 42;

  // add an array that is stored as std::vector (using an initializer list)
  j["list"] = {1, 0, 2};

  // add another object (using an initializer list of pairs)
  j["object"] = {{"currency", "USD"}, {"value", 42.99}};

  std::ostringstream oss;
  oss << std::setw(4) << j;
  std::string s = oss.str();
  std::cout << s << std::endl;

  nlohmann::json data = nlohmann::json::parse(s);

  std::cout << data << std::endl;

  std::cout << data["happy"] << std::endl;
  arma::rowvec restored_a = rowvec_from_json(data["happy"]);

  bool success = arma::all(restored_a == a);
  std::cout << "Perfect restoration: " << success << std::endl;
  return !success;
}