# Benchmark executables for libKriging
# These benchmarks test performance of Kriging operations with various data sizes

message(STATUS "Configuring benchmark executables")

# Create benchmark executables
add_executable(bench-kriging bench-kriging.cpp)
target_link_libraries(bench-kriging LINK_PUBLIC Kriging)

add_executable(bench-nugget-kriging bench-nugget-kriging.cpp)
target_link_libraries(bench-nugget-kriging LINK_PUBLIC Kriging)

add_executable(bench-noise-kriging bench-noise-kriging.cpp)
target_link_libraries(bench-noise-kriging LINK_PUBLIC Kriging)

# Optionally install benchmarks
install(TARGETS bench-kriging bench-nugget-kriging bench-noise-kriging
        RUNTIME DESTINATION bin/bench)

# Add custom target to build all benchmarks
add_custom_target(all_benchmarks
    DEPENDS bench-kriging bench-nugget-kriging bench-noise-kriging
    COMMENT "Building all benchmark executables")

# Add CTest entries for benchmarks (optional - can be run manually or via CI)
if(ENABLE_BENCH_TESTS)
    add_test(NAME bench-kriging-quick
            COMMAND bench-kriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(NAME bench-nugget-kriging-quick
            COMMAND bench-nugget-kriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(NAME bench-noise-kriging-quick
            COMMAND bench-noise-kriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_tests_properties(bench-kriging-quick bench-nugget-kriging-quick bench-noise-kriging-quick
            PROPERTIES LABELS "benchmark")
endif()

message(STATUS "Benchmark executables configured successfully")
