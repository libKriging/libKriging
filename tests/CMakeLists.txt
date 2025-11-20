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

add_executable(KrigingFitTest KrigingFitTest.cpp)
target_link_libraries(KrigingFitTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingFitTest)
catch_discover_tests(KrigingFitTest)

add_executable(KrigingUpdateTest KrigingUpdateTest.cpp)
target_link_libraries(KrigingUpdateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingUpdateTest)
catch_discover_tests(KrigingUpdateTest)

add_executable(KrigingUpdateSimulateTest KrigingUpdateSimulateTest.cpp)
target_link_libraries(KrigingUpdateSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingUpdateSimulateTest)
catch_discover_tests(KrigingUpdateSimulateTest)

add_executable(KrigingPredictTest KrigingPredictTest.cpp)
target_link_libraries(KrigingPredictTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingPredictTest)
catch_discover_tests(KrigingPredictTest)

add_executable(KrigingSimulateTest KrigingSimulateTest.cpp)
target_link_libraries(KrigingSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingSimulateTest)
catch_discover_tests(KrigingSimulateTest)

add_executable(NuggetKrigingTest NuggetKrigingTest.cpp)
target_link_libraries(NuggetKrigingTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingTest)
set(AdditionalCatchParameters --benchmark-samples 10)
catch_discover_tests(NuggetKrigingTest)
unset(AdditionalCatchParameters)

add_executable(NuggetKrigingFitTest NuggetKrigingFitTest.cpp)
target_link_libraries(NuggetKrigingFitTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingFitTest)
catch_discover_tests(NuggetKrigingFitTest)

add_executable(NuggetKrigingUpdateTest NuggetKrigingUpdateTest.cpp)
target_link_libraries(NuggetKrigingUpdateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingUpdateTest)
catch_discover_tests(NuggetKrigingUpdateTest)

add_executable(NuggetKrigingUpdateSimulateTest NuggetKrigingUpdateSimulateTest.cpp)
target_link_libraries(NuggetKrigingUpdateSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingUpdateSimulateTest)
catch_discover_tests(NuggetKrigingUpdateSimulateTest)

add_executable(NuggetKrigingPredictTest NuggetKrigingPredictTest.cpp)
target_link_libraries(NuggetKrigingPredictTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingPredictTest)
catch_discover_tests(NuggetKrigingPredictTest)

add_executable(NuggetKrigingSimulateTest NuggetKrigingSimulateTest.cpp)
target_link_libraries(NuggetKrigingSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingSimulateTest)
catch_discover_tests(NuggetKrigingSimulateTest)

add_executable(NoiseKrigingTest NoiseKrigingTest.cpp)
target_link_libraries(NoiseKrigingTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingTest)
set(AdditionalCatchParameters --benchmark-samples 10)
catch_discover_tests(NoiseKrigingTest)
unset(AdditionalCatchParameters)

add_executable(NoiseKrigingFitTest NoiseKrigingFitTest.cpp)
target_link_libraries(NoiseKrigingFitTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingFitTest)
catch_discover_tests(NoiseKrigingFitTest)

add_executable(NoiseKrigingUpdateTest NoiseKrigingUpdateTest.cpp)
target_link_libraries(NoiseKrigingUpdateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingUpdateTest)
catch_discover_tests(NoiseKrigingUpdateTest)

add_executable(NoiseKrigingUpdateSimulateTest NoiseKrigingUpdateSimulateTest.cpp)
target_link_libraries(NoiseKrigingUpdateSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingUpdateSimulateTest)
catch_discover_tests(NoiseKrigingUpdateSimulateTest)

add_executable(NoiseKrigingPredictTest NoiseKrigingPredictTest.cpp)
target_link_libraries(NoiseKrigingPredictTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingPredictTest)
catch_discover_tests(NoiseKrigingPredictTest)

add_executable(NoiseKrigingSimulateTest NoiseKrigingSimulateTest.cpp)
target_link_libraries(NoiseKrigingSimulateTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingSimulateTest)
catch_discover_tests(NoiseKrigingSimulateTest)

add_executable(KrigingLogLikTest KrigingLogLikTest.cpp)
target_link_libraries(KrigingLogLikTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KrigingLogLikTest)
catch_discover_tests(KrigingLogLikTest)

add_executable(NuggetKrigingLogLikTest NuggetKrigingLogLikTest.cpp)
target_link_libraries(NuggetKrigingLogLikTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NuggetKrigingLogLikTest)
catch_discover_tests(NuggetKrigingLogLikTest)

add_executable(NoiseKrigingLogLikTest NoiseKrigingLogLikTest.cpp)
target_link_libraries(NoiseKrigingLogLikTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NoiseKrigingLogLikTest)
catch_discover_tests(NoiseKrigingLogLikTest)

add_executable(LinearAlgebraTest LinearAlgebraTest.cpp)
target_link_libraries(LinearAlgebraTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries LinearAlgebraTest)
catch_discover_tests(LinearAlgebraTest)

add_executable(GetVersion GetVersion.cpp)
target_link_libraries(GetVersion LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries GetVersion)
add_test(NAME GetVersion COMMAND GetVersion)

add_executable(JsonTest demo/JsonTest.cpp)
target_link_libraries(JsonTest LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries JsonTest)
add_test(NAME JsonTest COMMAND JsonTest)

# example of callgrind command line
# valgrind --tool=callgrind --dump-instr=yes --trace-jump=yes -v ./tests/KrigingTest "logLikelihood benchmark" --benchmark-samples 10
# catch2 commande line options: https://github.com/catchorg/Catch2/blob/master/docs/command-line.md#specifying-which-tests-to-run

# list of all tests
# get_directory_property(all_tests TESTS)
add_executable(KSTestValidation KSTestValidation.cpp)
target_link_libraries(KSTestValidation LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries KSTestValidation)
catch_discover_tests(KSTestValidation)
