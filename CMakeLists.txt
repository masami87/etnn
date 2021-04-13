cmake_minimum_required(VERSION 3.17.0)
project(etnn CXX CUDA)

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags: Global
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -std=c++11 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -std=c++11 -O3")

include_directories(${PROJECT_SOURCE_DIR}/include 
                    ${PROJECT_SOURCE_DIR}/ps-lite/include)

# Build library
add_subdirectory(ps-lite)
add_subdirectory(src)

set(BUILD_TESTS ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

set(BUILD_EXAMPLES ON)
# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()