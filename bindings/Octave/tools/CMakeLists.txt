add_library(OctaveShared
        NonCopyable.hpp
        string_hash.hpp
        ObjectCollector.cpp ObjectCollector.hpp
        relative_error.cpp relative_error.hpp
        ObjectAccessor.cpp ObjectAccessor.hpp
        RequiresArg.cpp RequiresArg.hpp
        mx_accessor.hpp
        overload.hpp
        MxMapper.cpp MxMapper.hpp
        MxException.hpp 
        formatString.hpp)

target_link_libraries(OctaveShared PRIVATE armadillo Kriging)

set_target_properties(OctaveShared PROPERTIES
        COMPILE_FLAGS "${OCT_CPPFLAGS} ${OCT_CXXPICFLAGS} ${OCT_CXXFLAGS}")

#if (CXX_CLANG_TIDY)
#    set_target_properties(OctaveShared
#            PROPERTIES
#            CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
#endif ()
