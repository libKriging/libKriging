add_executable (tests main.cpp)

# Link the executable to the Kriging library.
# Since the Hello library has public include directories we will use those link directories when building
target_link_libraries (tests LINK_PUBLIC Kriging)
