#include "DictTest.hpp"

#include <pybind11/stl.h>
#include <iostream>
#include <typeinfo>

#include "../../../Octave/tools/formatString.hpp"
#include "../../../Octave/tools/string_hash.hpp"
#include "cast.hpp"

template <typename T>
bool check_dict_entryT(const py::dict& dict, std::string_view name) {
  auto x = get_entry<T>(dict, name);
  if (x.has_value()) {
    std::cout << "Entry '" << name << "' with type '" << CastSupportedType<T>::name << "' exists\n";
    std::cout << *x << '\n';
    return true;
  } else {
    std::cout << "Entry '" << name << "' does not exist\n";
    return false;
  }
}

bool check_dict_entry(const py::dict& dict, std::string_view name, std::string_view type_name) {
  switch (fnv_hash(type_name.data())) {
    case "bool"_hash:
      return check_dict_entryT<bool>(dict, name);
    case "int"_hash:
      return check_dict_entryT<int>(dict, name);
    case "float"_hash:
      return check_dict_entryT<double>(dict, name);
    case "str"_hash:
      return check_dict_entryT<std::string>(dict, name);
    case "vec"_hash:
      return check_dict_entryT<arma::vec>(dict, name);
    case "colvec"_hash:
      return check_dict_entryT<arma::colvec>(dict, name);
    case "rowvec"_hash:
      return check_dict_entryT<arma::rowvec>(dict, name);
    case "mat"_hash:
      return check_dict_entryT<arma::mat>(dict, name);
    default:
      throw std::runtime_error(formatString("type ", type_name, " is not (yet) managed with pylibkriging"));
  }
}