cmake_minimum_required(VERSION 3.14)

# VERSION must follow pattern : <major>.<minor>.<patch>
set(KRIGING_VERSION_MAJOR 0)
set(KRIGING_VERSION_MINOR 0)
set(KRIGING_VERSION_PATCH 0)
set(KRIGING_VERSION "${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}")

project(libKriging
        VERSION ${KRIGING_VERSION}
        DESCRIPTION "LibKriging")
# PORJECT_VERSION now contains also KRIGING_VERSION

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)