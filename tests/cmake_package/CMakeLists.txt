# Consumer project used by the `cmake_package` test: it only relies on the installed CMake package.
cmake_minimum_required(VERSION 3.13)
project(libKrigingPackageConsumer CXX)

find_package(libKriging ${EXPECTED_VERSION} CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE libKriging::Kriging)
