cmake_minimum_required(VERSION 3.13)
project(armadillo_lib_extractor LANGUAGES CXX)

include(${LIBKRIGING_PATH}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake)
get_target_property(arma_libs armadillo INTERFACE_LINK_LIBRARIES)

include(${LIBKRIGING_PATH}/lib/cmake/lbfgsb_cpp/lbfgsb_cppConfig.cmake)
get_target_property(lbfgsb_cpp_libs lbfgsb_cpp::lbfgsb_cpp INTERFACE_LINK_LIBRARIES)

# Extract OpenMP library if it was used during build
find_package(OpenMP)
set(openmp_libs)
if(OpenMP_CXX_FOUND AND OpenMP_CXX_LIBRARIES)
    set(openmp_libs "${OpenMP_CXX_LIBRARIES}")
endif()

# GPU backend runtime libraries (e.g. the static CUDA runtime), written by
# src/lib/CMakeLists.txt at libKriging build time; empty or absent when
# libKriging was built without a GPU backend.
set(gpu_libs)
if(EXISTS "${LIBKRIGING_PATH}/share/libKriging/libKriging_gpu_link_libs.txt")
    file(STRINGS "${LIBKRIGING_PATH}/share/libKriging/libKriging_gpu_link_libs.txt" gpu_libs)
endif()

foreach(lib IN LISTS arma_libs lbfgsb_cpp_libs openmp_libs gpu_libs)
    if(lib)
        message(STATUS "EXTERNAL_LIBS=${lib}")
    endif()
endforeach ()