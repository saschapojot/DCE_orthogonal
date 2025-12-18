cmake_minimum_required(VERSION 3.28)

project(DCE_quasilinear)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_LIB "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

######################## run_evolution part
add_executable(run_evolution run_evolution.cpp evolution/evolution.cpp evolution/evolution.hpp)
#add_executable(example fftw_example.cpp)
target_compile_options(run_evolution PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)
#target_compile_options(example PRIVATE
#        $<$<CONFIG:Release>:-O3>
#        $<$<CONFIG:Release>:-march=native>
#        $<$<CONFIG:Release>:-ftree-vectorize>
#)
#set_source_files_properties(
#        fftw_example.cpp
#        PROPERTIES
#
#        COMPILE_OPTIONS "-O3"
#)

set_source_files_properties(
        evolution/evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

set_source_files_properties(
        run_evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

set(Python3_INCLUDE_DIRS "/home/adada/anaconda3/include/python3.13")
set(Python3_LIBRARIES "/home/adada/anaconda3/lib/libpython3.13.so")
include_directories(${Python3_INCLUDE_DIRS})

set(Boost_NO_SYSTEM_PATHS YES)
if (Boost_NO_SYSTEM_PATHS)
    set(Boost_ROOT "/home/adada/cppLocal/boostBuild")
    set(Boost_INCLUDE_DIR "/home/adada/cppLocal/boostBuild/include")
    set(Boost_LIBRARY_DIR "/home/adada/cppLocal/boostBuild/lib")
endif (Boost_NO_SYSTEM_PATHS)

find_package(Boost REQUIRED COMPONENTS python numpy filesystem)

if(Boost_FOUND)
    message(STATUS "boost root is: ${Boost_ROOT}")
    message(STATUS "boost include path is : ${Boost_INCLUDE_DIR}")
    message(STATUS "boost library path is : ${Boost_LIBRARY_DIR}")
    message(STATUS "boost libraries is : ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
else()
    message(WARNING "boost not found.")
endif()


target_link_libraries(run_evolution Boost::filesystem)

target_link_libraries(run_evolution Boost::python)
target_link_libraries(run_evolution Boost::numpy)
#python
target_link_libraries(run_evolution ${Python3_LIBRARIES})

#include_directories("/home/adada/cppLocal/fftw3/include")
include_directories("/home/adada/cppLocal/armadillo/include")
#armadillo
file(GLOB LIBS_SO "/home/adada/cppLocal/armadillo/lib/libarmadillo.so")
target_link_libraries(run_evolution ${LIBS_SO})


#fftw3
#file(GLOB LIBS_A "/home/adada/cppLocal/fftw3/lib/libfftw3.a")
#target_link_libraries(run_evolution ${LIBS_A})



##### example
#target_link_libraries(example Boost::filesystem)
#
#target_link_libraries(example Boost::python)
#target_link_libraries(example Boost::numpy)
##python
#target_link_libraries(example ${Python3_LIBRARIES})


#armadillo
#file(GLOB LIBS_SO "/home/adada/cppLocal/armadillo/lib/libarmadillo.so")
#target_link_libraries(example ${LIBS_SO})


#fftw3
#file(GLOB LIBS_A "/home/adada/cppLocal/fftw3/lib/*.a")

#target_link_libraries(example ${LIBS_A})
########################end run_evolution part
add_executable(run_integral_table run_integral_table.cpp compute_integral_table/integral_table.cpp compute_integral_table/integral_table.hpp)

target_compile_options(run_integral_table PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)

set_source_files_properties(
        compute_integral_table/integral_table.cpp
        PROPERTIES
        COMPILE_OPTIONS "-O3"
)

target_link_libraries(run_integral_table Boost::filesystem)

target_link_libraries(run_integral_table Boost::python)
target_link_libraries(run_integral_table Boost::numpy)
#python
target_link_libraries(run_integral_table ${Python3_LIBRARIES})
target_link_libraries(run_integral_table  Boost::boost)

###gmp, mprf, flint
set(GMP_ROOT   "/home/adada/cppLocal/GMP")
set(MPFR_ROOT  "/home/adada/cppLocal/MPFR")
set(FLINT_ROOT "/home/adada/cppLocal/FLINT")

# -----------------------------------------------------------------------------
# Find GMP
# -----------------------------------------------------------------------------
find_path(GMP_INCLUDE_DIR gmp.h
        HINTS ${GMP_ROOT}/include
        NO_DEFAULT_PATH
)

find_library(GMP_LIBRARY gmp
        HINTS ${GMP_ROOT}/lib
        NO_DEFAULT_PATH
)

if (GMP_INCLUDE_DIR AND GMP_LIBRARY)
    message(STATUS "Found GMP include: ${GMP_INCLUDE_DIR}")
    message(STATUS "Found GMP lib: ${GMP_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find GMP in ${GMP_ROOT}")
endif()

# -----------------------------------------------------------------------------
# Find MPFR
# -----------------------------------------------------------------------------

find_path(MPFR_INCLUDE_DIR mpfr.h
        HINTS ${MPFR_ROOT}/include
        NO_DEFAULT_PATH
)
find_library(MPFR_LIBRARY mpfr
        HINTS ${MPFR_ROOT}/lib
        NO_DEFAULT_PATH
)

if (MPFR_INCLUDE_DIR AND MPFR_LIBRARY)
    message(STATUS "Found MPFR include: ${MPFR_INCLUDE_DIR}")
    message(STATUS "Found MPFR lib: ${MPFR_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find MPFR in ${MPFR_ROOT}")
endif()

# -----------------------------------------------------------------------------
# Find FLINT
# -----------------------------------------------------------------------------
# Note: FLINT headers are often inside a 'flint' subdirectory (e.g., include/flint/flint.h)
# or directly in include. We search for the file "flint.h".

find_path(FLINT_INCLUDE_DIR flint.h
        HINTS ${FLINT_ROOT}/include ${FLINT_ROOT}/include/flint
        NO_DEFAULT_PATH
)
find_library(FLINT_LIBRARY flint
        HINTS ${FLINT_ROOT}/lib
        NO_DEFAULT_PATH
)


if (FLINT_INCLUDE_DIR AND FLINT_LIBRARY)
    message(STATUS "Found FLINT include: ${FLINT_INCLUDE_DIR}")
    message(STATUS "Found FLINT lib: ${FLINT_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find FLINT in ${FLINT_ROOT}")
endif()
target_include_directories(
        run_integral_table
        PRIVATE
        ${FLINT_INCLUDE_DIR}
        ${MPFR_INCLUDE_DIR}
        ${GMP_INCLUDE_DIR}
)

target_link_libraries(run_integral_table
        ${FLINT_LIBRARY}
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
        m
)
