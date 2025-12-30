# Benchmark executables for libKriging
# These benchmarks test performance of Kriging operations with various data sizes

message(STATUS "Configuring benchmark executables")

# Create benchmark executables
add_executable(bench-kriging bench-kriging.cpp)
target_link_libraries(bench-kriging LINK_PUBLIC Kriging)

add_executable(bench-nuggetkriging bench-nuggetkriging.cpp)
target_link_libraries(bench-nuggetkriging LINK_PUBLIC Kriging)

add_executable(bench-noisekriging bench-noisekriging.cpp)
target_link_libraries(bench-noisekriging LINK_PUBLIC Kriging)

add_executable(bench-linearalgebra bench-linearalgebra.cpp)
target_link_libraries(bench-linearalgebra LINK_PUBLIC Kriging)

# Optionally install benchmarks
install(TARGETS bench-kriging bench-nuggetkriging bench-noisekriging bench-linearalgebra
        RUNTIME DESTINATION bin/bench)

# Add custom target to build all benchmarks
add_custom_target(all_benchmarks
    DEPENDS bench-kriging bench-nuggetkriging bench-noisekriging bench-linearalgebra
    COMMENT "Building all benchmark executables")

# Add CTest entries for benchmarks (optional - can be run manually or via CI)
if(ENABLE_BENCH_TESTS)
    add_test(NAME bench-kriging-quick
            COMMAND bench-kriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(NAME bench-nuggetkriging-quick
            COMMAND bench-nuggetkriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(NAME bench-noisekriging-quick
            COMMAND bench-noisekriging 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(NAME bench-linearalgebra-quick
            COMMAND bench-linearalgebra 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_tests_properties(bench-kriging-quick bench-nuggetkriging-quick bench-noisekriging-quick bench-linearalgebra-quick
            PROPERTIES LABELS "benchmark")
endif()

message(STATUS "Benchmark executables configured successfully")
