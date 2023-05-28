cmake_minimum_required(VERSION 3.5)
project(aca_openmp)

set(CMAKE_CXX_STANDARD 11)

# Set openmp compiler and linker flags
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(lud_seq main.cpp)
target_compile_definitions(lud_seq PUBLIC LUD_SEQ=1)

add_executable(lud_par main.cpp)
target_compile_definitions(lud_par PUBLIC LUD_PAR=1)

add_executable(lud_opt main.cpp)
target_compile_definitions(lud_opt PUBLIC LUD_OPT=1)