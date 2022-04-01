add_executable(manage_test_crash manage_test_crash.c)

add_mex_test(NAME success FILENAME dummy_succeededtest.m)
add_mex_test(NAME failure FILENAME dummy_failingtest.m WILL_FAIL)
add_mex_test(NAME mLibKriging-Demo FILENAME mLibKriging_demo.m)
add_mex_test(NAME mLibKriging-BadUsage FILENAME mLibKriging_failingtest.m WILL_FAIL)
add_mex_test(NAME LinearRegression-Simple FILENAME LinearRegression_simpletest.m)
add_mex_test(NAME LinearRegression-Complete FILENAME LinearRegression_completest.m)
add_mex_test(NAME BindingConsistency FILENAME binding_consistency_test.m)
