#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// doc : https://github.com/catchorg/Catch2/blob/master/docs/Readme.mda
// More example in https://github.com/catchorg/Catch2/tree/master/examples

TEST_CASE("Trival test", "[core]") {
  SECTION("checking trivial test ok") {
    REQUIRE((2 == 2));
  }
}

SCENARIO("vectors can be sized and resized", "[vector]") {
  GIVEN("A vector with some items") {
    std::vector<int> v(5);

    REQUIRE(v.size() == 5);
    REQUIRE(v.capacity() >= 5);

    WHEN("the size is increased") {
      v.resize(10);

      THEN("the size and capacity change") {
        REQUIRE(v.size() == 10);
        REQUIRE(v.capacity() >= 10);
      }
    }
    WHEN("the size is reduced") {
      v.resize(0);

      THEN("the size changes but not capacity") {
        REQUIRE(v.size() == 0);
        REQUIRE(v.capacity() >= 5);
      }
    }
    WHEN("more capacity is reserved") {
      v.reserve(10);

      THEN("the capacity changes but not the size") {
        REQUIRE(v.size() == 5);
        REQUIRE(v.capacity() >= 10);
      }
    }
    WHEN("less capacity is reserved") {
      v.reserve(0);

      THEN("neither size nor capacity are changed") {
        REQUIRE(v.size() == 5);
        REQUIRE(v.capacity() >= 5);
      }
    }
  }
}