include(CTest)
include(Catch)

# target_link_libraries links the executable to the library.
# Since the library has public include directories we will use those link directories when building

# Can be enabled if necessary
#if (CXX_CLANG_TIDY)
#    foreach (bin ${all_test_binaries})
#        set_target_properties(${bin}
#                PROPERTIES
#                CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
#    endforeach ()
#endif ()

if (LIBKRIGING_BENCHMARK_TESTS)
    message(STATUS "Benchmark tests are enabled")
    set(PARSE_CATCH_TESTS_NO_HIDDEN_TESTS OFF)
else ()
    message(STATUS "Benchmark tests are disabled; to enable them use -DLIBKRIGING_BENCHMARK_TESTS=ON option")
    set(PARSE_CATCH_TESTS_NO_HIDDEN_TESTS ON)
endif ()
#set(PARSE_CATCH_TESTS_VERBOSE ON)

# Build a list of test binaries
add_custom_target(all_test_binaries COMMENT "all unit tests.")

add_executable(class_unit_test demo/class_unit_test.cpp)
target_link_libraries(class_unit_test LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries class_unit_test)
add_test(NAME class_unit_test COMMAND class_unit_test)

add_executable(function_unit_test demo/function_unit_test.cpp)
target_link_libraries(function_unit_test LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries function_unit_test)
add_test(NAME function_unit_test COMMAND function_unit_test)

add_executable(armadillo_example demo/armadillo_example.cpp)
target_link_libraries(armadillo_example LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries armadillo_example)
add_test(NAME demo/armadillo_example COMMAND armadillo_example)

add_executable(catch2_unit_test catch2_unit_test.cpp)
target_link_libraries(catch2_unit_test LINK_PUBLIC Catch2)
add_dependencies(all_test_binaries catch2_unit_test)
catch_discover_tests(catch2_unit_test)

add_executable(regression_unit_test regression_unit_test.cpp)
target_link_libraries(regression_unit_test LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries regression_unit_test)
catch_discover_tests(regression_unit_test)

add_executable(KrigingTest KrigingTest.cpp)
target_link_libraries(KrigingTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingTest)
set(AdditionalCatchParameters --benchmark-samples 10)
catch_discover_tests(KrigingTest)
unset(AdditionalCatchParameters)

add_executable(CacheTest CacheFunctionTest.cpp)
target_link_libraries(CacheTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries CacheTest)
catch_discover_tests(CacheTest)

add_executable(GetVersion GetVersion.cpp)
target_link_libraries(GetVersion LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries GetVersion)
add_test(NAME GetVersion COMMAND GetVersion)

add_executable(JsonTest JsonTest.cpp)
target_link_libraries(JsonTest LINK_PUBLIC taocpp::json)
add_dependencies(all_test_binaries JsonTest)
add_test(NAME JsonTest COMMAND JsonTest)

# example of callgrind command line
# valgrind --tool=callgrind --dump-instr=yes --trace-jump=yes -v ./tests/KrigingTest "logLikelihood benchmark" --benchmark-samples 10
# catch2 commande line options: https://github.com/catchorg/Catch2/blob/master/docs/command-line.md#specifying-which-tests-to-run

# list of all tests
# get_directory_property(all_tests TESTS)
