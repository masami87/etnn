#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cstdio>
#include <iostream>
#include <memory>

#include "base.h"

// max thread num per block
#define MAX_THREAD 1024

#define getDims(A)                  \
    int dim_r, dim_c;               \
    dim_r = (A->size / MAX_THREAD); \
    if (dim_r == 0) {               \
        dim_r = 1;                  \
        dim_c = A->size;            \
    } else {                        \
        if (A->size % MAX_THREAD)   \
            dim_r++;                \
        dim_c = MAX_THREAD;         \
    }                               \
    dim3 dimGrid(dim_r);            \
    dim3 dimBlock(dim_c);

static int checkCudaError(cudaError_t code,
                          const char *expr,
                          const char *file,
                          int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'",
               file,
               line,
               (int)code,
               cudaGetErrorString(code),
               expr);
        return 1;
    }
    return 0;
}

static const char *cublasErrStr(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return NULL;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

static const char *curandErrStr(curandStatus_t error) {
    switch (error) {
        case CURAND_STATUS_SUCCESS:
            return NULL;

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define checkCublasErr(...)                               \
    do {                                                  \
        const char *err = cublasErrStr(__VA_ARGS__);      \
        if (err != NULL) {                                \
            printf("CUDA error at %s:%d, (%s) in '%s'\n", \
                   __FILE__,                              \
                   __LINE__,                              \
                   err,                                   \
                   #__VA_ARGS__);                         \
        }                                                 \
    } while (0)

#define checkCurandErr(...)                               \
    do {                                                  \
        const char *err = curandErrStr(__VA_ARGS__);      \
        if (err != NULL) {                                \
            printf("CUDA error at %s:%d, (%s) in '%s'\n", \
                   __FILE__,                              \
                   __LINE__,                              \
                   err,                                   \
                   #__VA_ARGS__);                         \
        }                                                 \
    } while (0)

#define checkCudaErr(...)                                                  \
    do {                                                                   \
        int err =                                                          \
            checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err)                                                           \
            abort();                                                       \
    } while (0)

static int checkCudnnError(cudnnStatus_t code,
                           const char *expr,
                           const char *file,
                           int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n",
               file,
               line,
               (int)code,
               cudnnGetErrorString(code),
               expr);
        return 1;
    }
    return 0;
}

#define checkCudnnErr(...)                                                  \
    do {                                                                    \
        int err =                                                           \
            checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err)                                                            \
            abort();                                                        \
    } while (0)


struct CudaContext {
    CudaContext() {
        checkCudnnErr(cudnnCreate(&m_cudnn));
        checkCublasErr(cublasCreate(&m_cublas));
    }
    ~CudaContext() {
        checkCudnnErr(cudnnDestroy(m_cudnn));
        checkCublasErr(cublasDestroy_v2(m_cublas));
    }

    cudnnHandle_t m_cudnn   = NULL;
    cublasHandle_t m_cublas = NULL;

};

static std::shared_ptr<CudaContext> createCudaContext() {
    return std::make_shared<CudaContext>();
}

static void setTensorDesc(cudnnTensorDescriptor_t &tensorDesc,
                          cudnnTensorFormat_t &tensorFormat,
                          cudnnDataType_t &dataType,
                          int n,
                          int c,
                          int h,
                          int w) {
    const int nDims    = 4;
    int dimA[nDims]    = {n, c, h, w};
    int strideA[nDims] = {c * h * w, h * w, w, 1};
    checkCudnnErr(
        cudnnSetTensorNdDescriptor(tensorDesc, dataType, 4, dimA, strideA));
}