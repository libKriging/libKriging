add_library(Kriging
        SHARED
        TestClass.cpp)

# Version on library name
set_target_properties(Kriging PROPERTIES VERSION ${KRIGING_VERSION})
# and a symlink with major number of version
set_target_properties(Kriging PROPERTIES SOVERSION ${KRIGING_VERSION_MAJOR})

# Declare public API of your library.
# This API will be installed for third-party application.
# It is a good practice to isolate it in your project tree (like placing it include/ directory).

# access to includes : https://cmake.org/cmake/help/latest/command/target_include_directories.html
## Pass top directory in included directories to avoid relative include path as ../lib/include/libKriging.h
## target_include_directories(Kriging PRIVATE .)
## target_include_directories (Kriging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Kriging PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/> # <prefix>/include/libKriging
        )

# Create install rule for your library. Use variables CMAKE_INSTALL_*DIR defined in GNUInstallDir
include(GNUInstallDirs)

# https://cmake.org/cmake/help/latest/command/install.html
install(TARGETS Kriging
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})