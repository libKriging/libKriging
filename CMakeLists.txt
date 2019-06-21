cmake_minimum_required(VERSION 3.11)

# CMake utils
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# VERSION must follow pattern : <major>.<minor>.<patch>
set(KRIGING_VERSION_MAJOR 0)
set(KRIGING_VERSION_MINOR 0)
set(KRIGING_VERSION_PATCH 0)
set(KRIGING_VERSION "${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}")

project(libKriging
        VERSION ${KRIGING_VERSION}
        DESCRIPTION "LibKriging")
# PORJECT_VERSION now contains also KRIGING_VERSION

#------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)

#------------------------------------------------------

add_subdirectory(src)
add_subdirectory(tests)
enable_testing()