#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// More example in https://github.com/catchorg/Catch2/tree/master/examples

TEST_CASE("Trival test", "[core]") {
    SECTION("checking trivial test ok") {
        REQUIRE((2 == 2));
    }
}