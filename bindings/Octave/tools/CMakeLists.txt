add_library(OctaveShared SHARED
        ObjectCollector.cpp ObjectCollector.hpp)

if (CXX_CLANG_TIDY)
    set_target_properties(OctaveShared
            PROPERTIES
            CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
endif ()
