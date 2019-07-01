enable_testing()

# May also be placed where binaries are built
add_custom_target(all_test_binaries COMMENT "all unit tests.")

add_dependencies(all_test_binaries class_unit_test)
add_test(NAME class_unit_test COMMAND class_unit_test)

add_dependencies(all_test_binaries function_unit_test)
add_test(NAME function_unit_test COMMAND function_unit_test)

add_dependencies(all_test_binaries armadillo_example)
add_test(NAME armadillo_example COMMAND armadillo_example)