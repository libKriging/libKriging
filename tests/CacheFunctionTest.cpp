#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "libKriging/CacheFunction.hpp"

TEST_CASE("Cache", "[core]") {
  auto env = 1;
  auto f = [env](double x) -> double { return env * x; };

  CacheFunction f_cached(f);
  
//  double x = 1;
//  REQUIRE(f_cached(x) == f(x));
//  REQUIRE(f_cached(x) == f(x));
}