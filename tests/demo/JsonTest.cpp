#include <iostream>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

int main(int argc, char** argv) {
  const int n = 200;

  arma::rowvec a(rand() % n, arma::fill::randn);
  arma::colvec b(rand() % n, arma::fill::randn);
  arma::mat c(rand() % n, rand() % n, arma::fill::randn);

  nlohmann::json j;

  // add a number that is stored as double (note the implicit conversion of j to an object)
  j["pi"] = 3.141;

  // add armadillo object stored as base64 encoded data
  j["rowvec"] = to_json(a);
  j["colvec"] = to_json(b);
  j["mat"] = to_json(c);

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

  std::cout << data["rowvec"] << std::endl;
  arma::rowvec restored_a = rowvec_from_json(data["rowvec"]);
  bool a_success = arma::all(restored_a == a);

  std::cout << data["colvec"] << std::endl;
  arma::colvec restored_b = colvec_from_json(data["colvec"]);
  bool b_success = arma::all(restored_b == b);

  std::cout << data["mat"] << std::endl;
  arma::mat restored_c = mat_from_json(data["mat"]);
  bool c_success = arma::all(arma::all(restored_c == c));

  bool success = a_success && b_success && c_success;

  std::cout << "Perfect restoration: " << success << std::endl;
  return !success;
}