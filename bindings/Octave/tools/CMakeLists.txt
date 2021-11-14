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
if (OctaveMode_INCLUDE_DIRS)
    target_include_directories(OctaveShared PUBLIC ${OctaveMode_INCLUDE_DIRS})
endif ()
if (OctaveMode_COMPILE_FLAGS)
    set_target_properties(OctaveShared PROPERTIES COMPILE_FLAGS ${OctaveMode_COMPILE_FLAGS})
endif ()

#if (CXX_CLANG_TIDY)
#    set_target_properties(OctaveShared
#            PROPERTIES
#            CXX_CLANG_TIDY ${CXX_CLANG_TIDY})
#endif ()
