include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)

add_mex_function(NAME mypow2 SOURCES mypow2.cpp LINK_LIBRARIES Kriging)