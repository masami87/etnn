#include "initializer.h"

__global__ void constant(float* a, float v, long int size) {
    long int thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id_x < size) {
        a[thread_id_x] = v;
    }
}


void IConstant::apply(FloatTensor* tensor) {
    getDims(tensor);
    constant<<<dimGrid, dimBlock>>>(tensor->data_ptr(), value, tensor->size);
    checkCudaErr(cudaDeviceSynchronize());
}