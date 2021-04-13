find_package(CUDAToolkit)


# C++ files
file(GLOB_RECURSE CPP_HEADERS "${PROJECT_SOURCE_DIR}/src/*" *.{h, hpp})
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cc, cpp})
file(GLOB_RECURSE CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cu})

SET(ALL_FILES ${PUBLIC_HEADERS} ${CPP_HEADERS} ${CPP_SOURCES} ${CUDA_SOURCES})


############################# SET LIBRARY #################################

set(LIBS pslite protobuf ${ZMQ_LIBRARIES})

add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/src
        )

set(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_CUDNN_LIBRARY "/usr/local/cuda/lib64/libcudnn.so")


list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_CUDNN_LIBRARY})


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Project name: " ${CMAKE_PROJECT_NAME} )
message(STATUS "Project version: " ${CMAKE_PROJECT_VERSION} )
message(STATUS "Prefix path: " ${CMAKE_PREFIX_PATH} )
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX} )
message(STATUS "-------------------------------------------" )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )

message(STATUS "-------------------------------------------" )
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}) | Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "C++ flags (release): " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "C++ flags (debug): " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "-------------------------------------------" )

message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})