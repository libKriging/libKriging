# Prepare Catch2 libray (header only)
target_include_directories(Catch2 INTERFACE ${CATCH_INCLUDE_DIR})

# target_link_libraries links the executable to the library.
# Since the library has public include directories we will use those link directories when building

# Build a list of test binaries
add_custom_target(all_test_binaries COMMENT "all unit tests.")

add_executable (class_unit_test class_unit_test.cpp)
target_link_libraries(class_unit_test LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries class_unit_test)

add_executable (function_unit_test function_unit_test.cpp)
target_link_libraries(function_unit_test LINK_PUBLIC Kriging)
add_dependencies(all_test_binaries function_unit_test)

add_executable(armadillo_example armadillo_example.cpp)
target_link_libraries(armadillo_example LINK_PUBLIC armadillo)
add_dependencies(all_test_binaries armadillo_example)

add_executable(catch2_unit_test catch2_unit_test.cpp)
target_link_libraries(catch2_unit_test LINK_PUBLIC Catch2)
add_dependencies(all_test_binaries catch2_unit_test)

enable_testing()

# Manual test
add_test(NAME class_unit_test COMMAND class_unit_test)
add_test(NAME function_unit_test COMMAND function_unit_test)
add_test(NAME armadillo_example COMMAND armadillo_example)

# Test using catch2 (parse catch tests)
ParseAndAddCatchTests(catch2_unit_test)