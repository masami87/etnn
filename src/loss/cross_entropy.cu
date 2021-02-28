#include "cross_entropy.h"

__global__ void cross_entropy_kernel(float* output,
                                     float* label,
                                     float* sum_array,
                                     int batch,
                                     int feature) {
    long int thread_id_x =
        blockIdx.x * blockDim.x + threadIdx.x;  // Batch index

    if (thread_id_x < batch) {
        float eps            = 1e-8;
        unsigned int batch_i = thread_id_x;  // Alias

        // Contiguous data
        unsigned int start = batch_i * feature;
        unsigned int end   = start + feature;

        // Compute cross-entropy
        float bi_sum = 0.0f;
        for (unsigned int i = start; i < end; i++) {
            bi_sum += label[i] * logf(output[i] + eps);
        }
        // Store partial sums (later will be reduced)
        sum_array[thread_id_x] = -bi_sum;
    }
}

__global__ void cross_entropy_d_kernel(float* output,
                                       float* label,
                                       float* diff,
                                       long int size) {
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;  //  index
    if (thread_id_x < size) {
        float eps = 1e-8;
        // diff[thread_id_x] =
        //      - label[thread_id_x];
    
        diff[thread_id_x] =
            -label[thread_id_x] * (1.0f / (output[thread_id_x] + eps));
    }
}

float CpuReductionSum(float* array, size_t n) {
    float sum = 0.f;
    for (size_t i = 0; i < n; i++) {
        sum += *(array + i);
    }
    return sum;
}

CrossEntropy::CrossEntropy()
    : Loss("cross_entropy") {
}
// cross entropy
// https://blog.csdn.net/jasonleesjtu/article/details/89426465

float CrossEntropy::compute(std::shared_ptr<FloatTensor> output,
                            std::shared_ptr<FloatTensor> label) {
    auto l_shape = label->shape;
    if(l_shape.size() == 2){
        l_shape = vector<int>{l_shape[0], l_shape[1], 1, 1};
    }

    CHECK(output->shape == l_shape);
    int batch   = output->shape[0];
    int feature = output->shape[1] * output->shape[2] * output->shape[3];
    float* sum_array;
    checkCudaErr(cudaMalloc((void**)&sum_array, batch * sizeof(float)));
    int blockSize = 1024;
    int gridSize  = (batch + blockSize - 1) / blockSize;
    cross_entropy_kernel<<<gridSize, blockSize>>>(
        output->data_ptr(), label->data_ptr(), sum_array, batch, feature);
    checkCudaErr(cudaDeviceSynchronize());
    float* sum = new float[batch * sizeof(float)];
    checkCudaErr(cudaMemcpy((void*)sum,
                            (void*)sum_array,
                            batch * sizeof(float),
                            cudaMemcpyDeviceToHost));
    checkCudaErr(cudaFree(sum_array));
    //TODO: write a gpu version
    float mean = CpuReductionSum(sum, batch) / batch;
    return mean;
}

void CrossEntropy::delta(std::shared_ptr<FloatTensor> output,
                         std::shared_ptr<FloatTensor> label,
                         std::shared_ptr<FloatTensor> diff) {
    auto l_shape = label->shape;
    if(l_shape.size() == 2){
        l_shape = vector<int>{l_shape[0], l_shape[1], 1, 1};
    }
    CHECK(output->shape == l_shape);
    CHECK(output->shape == diff->shape);
    getDims(output);
    int sz = output->size;
    cross_entropy_d_kernel<<<dimGrid, dimBlock>>>(
        output->data_ptr(), label->data_ptr(), diff->data_ptr(), sz);
    checkCudaErr(cudaDeviceSynchronize());
}