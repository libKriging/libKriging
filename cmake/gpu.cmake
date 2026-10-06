# Resolves the GPU backend options (see the root CMakeLists.txt) into the
# internal LIBKRIGING_{CUDA,HIP,SYCL,METAL}_ITERATIVE (ON/OFF) cache entries
# read by src/lib, bench and the bindings. AUTO never fails: a backend whose
# toolchain is missing is turned OFF with a one-line reason. ON keeps its
# historical meaning (required; src/lib fails loudly if unusable).

string(TOUPPER "${ENABLE_GPU_ITERATIVE}" _lk_gpu_all)
if (NOT _lk_gpu_all STREQUAL "" AND NOT _lk_gpu_all MATCHES "^(AUTO|OFF)$")
    logFatalError("Invalid ENABLE_GPU_ITERATIVE option '${ENABLE_GPU_ITERATIVE}'; choose between AUTO, OFF or empty.")
endif ()

# _lk_gpu_mode(<BACKEND> <out-var>): effective mode ON / OFF / AUTO
function(_lk_gpu_mode backend out)
    if (_lk_gpu_all STREQUAL "AUTO" OR _lk_gpu_all STREQUAL "OFF")
        set(mode "${_lk_gpu_all}")
    else ()
        string(TOUPPER "${ENABLE_${backend}_ITERATIVE}" mode)
        if (mode STREQUAL "AUTO")
        elseif (ENABLE_${backend}_ITERATIVE)  # ON/TRUE/1/YES...
            set(mode "ON")
        else ()
            set(mode "OFF")
        endif ()
    endif ()
    set(${out} "${mode}" PARENT_SCOPE)
endfunction()

# _lk_gpu_result(<BACKEND> <ON|OFF> <reason>)
macro(_lk_gpu_result backend value reason)
    set(LIBKRIGING_${backend}_ITERATIVE ${value} CACHE INTERNAL "Resolved ENABLE_${backend}_ITERATIVE (ON/OFF)" FORCE)
    message(STATUS "${backend} iterative backend: ${reason}")
endmacro()

include(CheckLanguage)

# --- CUDA ---------------------------------------------------------------
_lk_gpu_mode(CUDA _mode)
if (_mode STREQUAL "AUTO")
    if (${CMAKE_VERSION} VERSION_LESS "3.23")
        _lk_gpu_result(CUDA OFF "AUTO -> OFF (needs CMake >= 3.23, found ${CMAKE_VERSION})")
    else ()
        check_language(CUDA)
        # check_language() trusts a preset CMAKE_CUDA_COMPILER without
        # testing it: at least make sure it exists before relying on it.
        if (CMAKE_CUDA_COMPILER)
            find_program(_lk_cuda_compiler_path NAMES "${CMAKE_CUDA_COMPILER}" NO_CACHE)
            if (NOT _lk_cuda_compiler_path)
                set(CMAKE_CUDA_COMPILER "")
            endif ()
        endif ()
        if (CMAKE_CUDA_COMPILER)
            find_package(CUDAToolkit QUIET)
            if (TARGET CUDA::cudart_static)
                if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                    # Portable default: every major real architecture the
                    # toolkit supports plus PTX for the newest one (JIT on
                    # future GPUs). Unlike "native", needs no GPU at build time.
                    set(CMAKE_CUDA_ARCHITECTURES all-major)
                endif ()
                _lk_gpu_result(CUDA ON "AUTO -> ON (CUDA ${CUDAToolkit_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES})")
            else ()
                _lk_gpu_result(CUDA OFF "AUTO -> OFF (CUDA compiler found but no usable CUDA toolkit)")
            endif ()
        else ()
            _lk_gpu_result(CUDA OFF "AUTO -> OFF (no working CUDA compiler; set CMAKE_CUDA_COMPILER / CMAKE_CUDA_HOST_COMPILER to use one)")
        endif ()
    endif ()
else ()
    _lk_gpu_result(CUDA ${_mode} "${_mode}")
endif ()

# --- HIP (AMD ROCm) -----------------------------------------------------
_lk_gpu_mode(HIP _mode)
if (_mode STREQUAL "AUTO")
    if (${CMAKE_VERSION} VERSION_LESS "3.21")
        _lk_gpu_result(HIP OFF "AUTO -> OFF (needs CMake >= 3.21, found ${CMAKE_VERSION})")
    else ()
        check_language(HIP)
        if (CMAKE_HIP_COMPILER)
            find_program(_lk_hip_compiler_path NAMES "${CMAKE_HIP_COMPILER}" NO_CACHE)
            if (NOT _lk_hip_compiler_path)
                set(CMAKE_HIP_COMPILER "")
            endif ()
        endif ()
        if (CMAKE_HIP_COMPILER)
            find_package(hip QUIET)
        endif ()
        if (NOT CMAKE_HIP_COMPILER)
            _lk_gpu_result(HIP OFF "AUTO -> OFF (no working HIP compiler)")
        elseif (NOT hip_FOUND)
            _lk_gpu_result(HIP OFF "AUTO -> OFF (HIP compiler found but no hip-config.cmake; set CMAKE_PREFIX_PATH to the ROCm install)")
        elseif (DEFINED CMAKE_HIP_ARCHITECTURES)
            _lk_gpu_result(HIP ON "AUTO -> ON (architectures: ${CMAKE_HIP_ARCHITECTURES})")
        else ()
            # Unlike CUDA there is no portable "all" target list worth
            # compiling for, and "native" fails without a GPU: build for the
            # AMD GPUs actually present, and stay OFF when there is none.
            find_program(_lk_rocm_agent_enumerator rocm_agent_enumerator
                    HINTS ENV ROCM_PATH /opt/rocm PATH_SUFFIXES bin)
            set(_lk_hip_archs "")
            if (_lk_rocm_agent_enumerator)
                execute_process(COMMAND ${_lk_rocm_agent_enumerator}
                        OUTPUT_VARIABLE _lk_agents ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
                string(REGEX MATCHALL "gfx[0-9a-f]+" _lk_agents "${_lk_agents}")
                list(REMOVE_ITEM _lk_agents gfx000)  # the CPU agent
                list(REMOVE_DUPLICATES _lk_agents)
                set(_lk_hip_archs "${_lk_agents}")
            endif ()
            if (_lk_hip_archs)
                set(CMAKE_HIP_ARCHITECTURES "${_lk_hip_archs}" CACHE STRING "HIP architectures (detected)")
                _lk_gpu_result(HIP ON "AUTO -> ON (detected AMD GPU architectures: ${_lk_hip_archs})")
            else ()
                _lk_gpu_result(HIP OFF "AUTO -> OFF (ROCm found but no AMD GPU detected; set CMAKE_HIP_ARCHITECTURES to force)")
            endif ()
        endif ()
    endif ()
else ()
    _lk_gpu_result(HIP ${_mode} "${_mode}")
endif ()

# --- SYCL (Intel oneAPI) ------------------------------------------------
_lk_gpu_mode(SYCL _mode)
if (_mode STREQUAL "AUTO")
    include(CheckCXXSourceCompiles)
    set(_lk_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fsycl")
    check_cxx_source_compiles("#include <sycl/sycl.hpp>
int main() { sycl::queue q; return 0; }" LIBKRIGING_HAVE_SYCL)
    set(CMAKE_REQUIRED_FLAGS "${_lk_saved_flags}")
    if (LIBKRIGING_HAVE_SYCL)
        _lk_gpu_result(SYCL ON "AUTO -> ON (compiler accepts -fsycl; backend UNVERIFIED)")
    else ()
        _lk_gpu_result(SYCL OFF "AUTO -> OFF (C++ compiler does not accept -fsycl)")
    endif ()
else ()
    _lk_gpu_result(SYCL ${_mode} "${_mode}")
endif ()

# --- Metal (Apple) ------------------------------------------------------
_lk_gpu_mode(METAL _mode)
if (_mode STREQUAL "AUTO")
    if (NOT APPLE)
        _lk_gpu_result(METAL OFF "AUTO -> OFF (not macOS)")
    else ()
        if (NOT DEFINED METAL_CPP_DIR)
            find_path(_lk_metal_cpp_dir Metal/Metal.hpp
                    HINTS ENV METAL_CPP_DIR ${CMAKE_SOURCE_DIR}/.deps/metal-cpp)
            if (_lk_metal_cpp_dir)
                set(METAL_CPP_DIR "${_lk_metal_cpp_dir}" CACHE PATH "metal-cpp headers (detected)")
            endif ()
        endif ()
        if (DEFINED METAL_CPP_DIR)
            _lk_gpu_result(METAL ON "AUTO -> ON (metal-cpp: ${METAL_CPP_DIR}; backend UNVERIFIED)")
        else ()
            _lk_gpu_result(METAL OFF "AUTO -> OFF (metal-cpp headers not found; set METAL_CPP_DIR)")
        endif ()
    endif ()
else ()
    _lk_gpu_result(METAL ${_mode} "${_mode}")
endif ()
