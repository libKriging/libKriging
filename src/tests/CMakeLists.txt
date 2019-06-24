set(ALL_TESTS)

add_executable (class_unit_test class_unit_test.cpp)
list(APPEND ALL_TESTS class_unit_test)
# Link the executable to the Kriging library.
# Since the library has public include directories we will use those link directories when building
target_link_libraries (class_unit_test LINK_PUBLIC Kriging)

add_executable (function_unit_test function_unit_test.cpp)
list(APPEND ALL_TESTS function_unit_test)
target_link_libraries (function_unit_test LINK_PUBLIC Kriging)