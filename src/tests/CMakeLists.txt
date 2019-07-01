# target_link_libraries links the executable to the library.
# Since the library has public include directories we will use those link directories when building

add_executable (class_unit_test class_unit_test.cpp)
target_link_libraries (class_unit_test LINK_PUBLIC Kriging)

add_executable (function_unit_test function_unit_test.cpp)
target_link_libraries (function_unit_test LINK_PUBLIC Kriging)

add_executable(armadillo_example armadillo_example.cpp)
target_link_libraries (armadillo_example LINK_PUBLIC armadillo)
