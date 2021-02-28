#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#define M 2
#define N 4
#define K 3

#define batch 2
#define input 3
#define output 4

void printMatrix(float (*matrix)[N], int row, int col, bool reverse) {
    if (reverse) {
        int temp;
        temp = row;
        row  = col;
        col  = temp;
    }
    std::cout << " address of matrix: " << matrix
              << " address of (matrix+1):" << (matrix + 1)
              << " sizeof(matrix): " << sizeof(matrix)
              << " sizeof(*matrix): " << sizeof(*matrix) << std::endl;
    for (int i = 0; i < row; i++) {
        std::cout << std::endl;
        std::cout << " [ ";
        for (int j = 0; j < col; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << " ] ";
    }
    std::cout << std::endl;
}

int main(void) {
    float alpha              = 1.0;
    float beta               = 0.0;
    float h_A[batch][input]  = {{1, 2, 3}, {4, 5, 6}};  // data
    float h_B[output][input] = {
        {1, 2, 3}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}};  // param
    float h_C[batch][output] = {0};
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, batch * input * sizeof(float));
    cudaMalloc((void **)&d_b, output * input * sizeof(float));
    cudaMalloc((void **)&d_c, batch * output * sizeof(float));
    cudaMemcpy(
        d_a, &h_A, batch * input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_b, &h_B, output * input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, batch * output * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                output,
                batch,
                input,
                &alpha,
                d_b,
                input,
                d_a,
                input,
                &beta,
                d_c,
                output);
    cudaMemcpy(h_C,
               d_c,
               batch * output * sizeof(float),
               cudaMemcpyDeviceToHost);  //此处的h_C是按列存储的CT
    printMatrix(h_C, batch, output, false);  //按行读取h_C相当于做了CTT=C的结果
    return 0;
}