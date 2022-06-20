cmake_minimum_required(VERSION 3.13)
project(armadillo_lib_extractor LANGUAGES NONE)

include(${LIBKRIGING_PATH}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake)
get_target_property(arma_libs armadillo INTERFACE_LINK_LIBRARIES)

include(${LIBKRIGING_PATH}/lib/cmake/lbfgsb/lbfgsbConfig.cmake)
get_target_property(lbfgsb_libs lbfgsb::lbfgsb INTERFACE_LINK_LIBRARIES)

#include(${LIBKRIGING_PATH}/lib/cmake/lbfgsb_cpp/lbfgsb_cppConfig.cmake)
#get_target_property(lbfgsb_cpp_libs lbfgsb_cpp::lbfgsb_cpp INTERFACE_LINK_LIBRARIES)

set(external_libs ${arma_libs} ${lbfgsb_libs} ${lbfgsb_cpp_libs})
list(JOIN external_libs " " lib_str)
message(STATUS EXTERNAL_LIBS=${lib_str})
