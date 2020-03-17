#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// doc : https://github.com/catchorg/Catch2/blob/master/docs/Readme.mda
// More example in https://github.com/catchorg/Catch2/tree/master/examples

#include <libKriging/utils/enum_utils.hpp>

enum class TestEnum { A, B, C, D };

template <>
char const* enumStrings<TestEnum>::data[] = {"a", "b", "c", "d"};

TEST_CASE("enum conversion by reflection", "") {
  REQUIRE(enum_count<TestEnum>() == 4);
  REQUIRE(enumToString(TestEnum::A) == "a");
  REQUIRE(enumToString(TestEnum::B) == "b");
  REQUIRE(enumToString(TestEnum::C) == "c");
  REQUIRE(enumToString(TestEnum::D) == "d");
  REQUIRE(enumFromString<TestEnum>("a") == TestEnum::A);
  REQUIRE(enumFromString<TestEnum>("b") == TestEnum::B);
  REQUIRE(enumFromString<TestEnum>("c") == TestEnum::C);
  REQUIRE(enumFromString<TestEnum>("d") == TestEnum::D);
}