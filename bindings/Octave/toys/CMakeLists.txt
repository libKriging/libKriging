add_library(OctaveToys
        func1.cpp func1.hpp
        func_new.cpp func_new.hpp
        relative_error.cpp relative_error.hpp)

set_target_properties(OctaveToys PROPERTIES
        COMPILE_FLAGS "${OCT_CPPFLAGS} ${OCT_CXXPICFLAGS} ${OCT_CXXFLAGS}")

target_link_libraries(OctaveToys Kriging)

#if (CXX_CLANG_TIDY)
#    set_target_properties(OctaveToys
#            PROPERTIES
#            CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
#endif ()
