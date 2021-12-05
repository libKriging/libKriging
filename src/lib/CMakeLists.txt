# SHARED or STATIC type if defined using BUILD_SHARED_LIBS variable
# ex: cmake -DBUILD_SHARED_LIBS:BOOL=OFF
# default value is defined in root CMakeLists.txt
add_library(Kriging
        demo/DemoClass.cpp include/libKriging/demo/DemoClass.hpp
        demo/DemoFunction.cpp include/libKriging/demo/DemoFunction.hpp
        demo/DemoArmadilloClass.cpp include/libKriging/demo/DemoArmadilloClass.hpp
        LinearRegression.cpp include/libKriging/LinearRegression.hpp
        LinearRegressionOptim.cpp include/libKriging/LinearRegressionOptim.hpp
        Kriging.cpp include/libKriging/Kriging.hpp
        include/libKriging/KrigingException.hpp
        lkalloc.cpp include/libKriging/utils/lkalloc.hpp
        Bench.cpp include/libKriging/Bench.hpp)

if (CXX_CLANG_TIDY)
    set_target_properties(Kriging
            PROPERTIES
            CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
endif ()

target_link_libraries(Kriging PUBLIC armadillo)

# TODO check if this method is useful
# * https://atomheartother.github.io/c++/2018/07/12/CPPDynLib.html
#   To build shared libraries in Windows, we set CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS to TRUE.
#   See https://cmake.org/cmake/help/v3.11/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
#   See https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
#   set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# https://cmake.org/cmake/help/v3.11/module/GenerateExportHeader.html
include (GenerateExportHeader)
GENERATE_EXPORT_HEADER(Kriging # generates the export header `lib`_EXPORTS.h automatically
        BASE_NAME LIBKRIGING
        EXPORT_FILE_NAME libKriging/libKriging_exports.h)

# Version on library name
set_target_properties(Kriging PROPERTIES VERSION ${KRIGING_VERSION})
# and a symlink with major number of version
set_target_properties(Kriging PROPERTIES SOVERSION ${KRIGING_VERSION_MAJOR})

# default visibility is hidden for Kriging lib
set_target_properties(Kriging PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Declare public API of your library.
# This API will be installed for third-party application.
# It is a good practice to isolate it in your project tree (like placing it include/ directory).

# access to includes : https://cmake.org/cmake/help/latest/command/target_include_directories.html
## Pass top directory in included directories to avoid relative include path as ../lib/include/libKriging.h
## target_include_directories(Kriging PRIVATE .)
## target_include_directories (Kriging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Kriging
        PUBLIC
            # path of the headers after installation
            $<INSTALL_INTERFACE:include> # <prefix>/include/libKriging
            # path of the headers before installation
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            # path of the generated headers before installation
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

add_subdirectory(include)

# Create install rule for your library. Use variables CMAKE_INSTALL_*DIR defined in GNUInstallDir
include(GNUInstallDirs)
# https://cmake.org/cmake/help/latest/command/install.html
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/
install(TARGETS Kriging
        EXPORT libKriging-export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel
        )
install(EXPORT libKriging-export
        FILE libKriging.cmake
        NAMESPACE libKriging::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
        )
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/libKriging/libKriging_exports.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libKriging
        )
