#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "libKriging/CacheFunction.hpp"

TEST_CASE("Cache of 0-arg function", "[core]") {
  auto f = []() -> double { return 1; };

  CacheFunction f_cached(f);

  REQUIRE(f_cached.inspect() == 0);
  REQUIRE(f_cached() == f());
  REQUIRE(f_cached.inspect() == 1);
  REQUIRE(f_cached() == f());
  REQUIRE(f_cached.inspect() == 2);

  auto stat = f_cached.stat();
  REQUIRE(stat.min_hit == 2);
  REQUIRE(stat.total_hit == 2);
  REQUIRE(stat.cache_size == 1);
}

TEST_CASE("Cache of 1-arg function", "[core]") {
  auto f = [](double x) -> double { return x; };

  SECTION("Same arg should increase hint count") {
    CacheFunction f_cached(f);

    double x = 1;
    REQUIRE(f_cached.inspect(x) == 0);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 1);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 2);

    auto stat = f_cached.stat();
    REQUIRE(stat.min_hit == 2);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 1);
  }

  SECTION("Different args should not increase hint count") {
    CacheFunction f_cached(f);

    double x = 1;
    REQUIRE(f_cached.inspect(x) == 0);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 1);
    x = 2;
    REQUIRE(f_cached.inspect(x) == 0);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 1);

    auto stat = f_cached.stat();
    REQUIRE(stat.min_hit == 1);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 2);
  }
}

TEST_CASE("Cache of 2-args function", "[core]") {
  auto f = [](double x, double y) -> double { return x + y; };

  SECTION("Same arg should increase hint count") {
    CacheFunction f_cached(f);

    double x = 1, y = 2;
    REQUIRE(f_cached.inspect(x, y) == 0);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 1);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 2);

    auto stat = f_cached.stat();
    REQUIRE(stat.min_hit == 2);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 1);
  }

  SECTION("Different args should not increase hint count") {
    CacheFunction f_cached(f);

    double x = 1, y = 2;
    REQUIRE(f_cached.inspect(x, y) == 0);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 1);
    x = 2;  // different order should be a different value
    y = 1;
    REQUIRE(f_cached.inspect(x, y) == 0);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 1);

    auto stat = f_cached.stat();
    REQUIRE(stat.min_hit == 1);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 2);
  }
}

TEST_CASE("Cache of 0-arg function with global context", "[core]") {
  double context = 1;
  auto f = [&context]() -> double { return context; };

  SECTION("Global context change should increase hit count") {
    CacheFunction f_cached(f, context);

    REQUIRE(f_cached.inspect() == 0);
    REQUIRE(f_cached() == f());
    REQUIRE(f_cached.inspect() == 1);
    context = 2;
    REQUIRE(f_cached.inspect() == 0);
    REQUIRE(f_cached() == f());
    REQUIRE(f_cached.inspect() == 1);

    auto stat = f_cached.stat();
    REQUIRE(stat.max_hit == 1);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 2);
  }
}

TEST_CASE("Cache of 1-arg function with global context", "[core]") {
  double context = 1;
  auto f = [&context](double x) -> double { return x * context; };

  SECTION("Global context change should increase hint count") {
    CacheFunction f_cached(f, context);

    double x = 1;
    REQUIRE(f_cached.inspect(x) == 0);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 1);
    context = 2;
    REQUIRE(f_cached.inspect(x) == 0);
    REQUIRE(f_cached(x) == f(x));
    REQUIRE(f_cached.inspect(x) == 1);

    auto stat = f_cached.stat();
    REQUIRE(stat.max_hit == 1);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 2);
  }
}

TEST_CASE("Cache of 2-args function with global context", "[core]") {
  double context = 1;
  auto f = [&context](double x, double y) -> double { return (x + y) * context; };

  SECTION("Global context change should increase hint count") {
    CacheFunction f_cached(f, context);

    double x = 1, y = 2;
    REQUIRE(f_cached.inspect(x, y) == 0);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 1);
    context = 2;
    REQUIRE(f_cached.inspect(x, y) == 0);
    REQUIRE(f_cached(x, y) == f(x, y));
    REQUIRE(f_cached.inspect(x, y) == 1);

    auto stat = f_cached.stat();
    REQUIRE(stat.max_hit == 1);
    REQUIRE(stat.total_hit == 2);
    REQUIRE(stat.cache_size == 2);
  }
}