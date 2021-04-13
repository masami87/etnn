#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "base.h"
#include "device.h"

using std::vector;

class Tensor;

class CudaTensor;
using FloatTensor = CudaTensor;

class Tensor {
 public:
    int isgpu              = false;
    unsigned int dim       = 0;
    unsigned long int size = 0;
    vector<int> shape;
    vector<int> stride;

    Tensor() = delete;
    Tensor(const vector<int> &shape)
        : Tensor(shape, nullptr, false) {
        data = new float[size]();
    }
    Tensor(const vector<int> &shape, float *ptr, bool isgpu = false)
        : shape(shape)
        , data(ptr)
        , isgpu(isgpu) {
        Reshape(shape);
    }

    Tensor(Tensor &&other) {  // only allow move semantic
        this->data  = other.data;
        other.data  = nullptr;
        this->isgpu = other.isgpu;
        this->shape = other.shape;
        Reshape(this->shape);
    }

    virtual ~Tensor() {
        if (data)
            delete[] data;
        data = nullptr;
    };

    void Reshape(const vector<int> &shape) {
        this->shape = shape;
        CHECK(shape.size() > 0) << "(Tensor dim must be non zero!)";
        this->dim = shape.size();

        // update size and stride
        this->stride     = vector<int>(dim, 0);
        unsigned long sz = 1;
        for (int i = dim - 1; i >= 0; i--) {
            stride[i] = sz;
            sz *= shape[i];
        }
        CHECK(sz > 0) << "(Tensor shape must be non zero!)";
        if (this->size != 0) {
            CHECK(this->size == sz) << "(Reshape cannot change the size of "
                                       "tensor!)";
        }
        this->size = sz;
    }

    float &value(const vector<int> &inx) {
        CHECK(inx.size() == dim);
        int offset = 0;
        for (int i = 0; i < dim; i++) {
            offset += inx[i] * stride[i];
        }
        return data[offset];
    }

    int get_size() const {
        return size;
    }

    float *data_ptr() const {
        return data;
    }
    float *data_ptr() {
        return data;
    }

 protected:
    float *data;
};

class CudaTensor final : public Tensor {
 public:
    CudaTensor() = delete;

    CudaTensor(const vector<int> &shape)
        : Tensor(shape, nullptr, true) {
        checkCudaErr(cudaMalloc(&data, size * sizeof(float)));
    }

    CudaTensor(vector<int> &&shape)
        : Tensor(shape, nullptr, true) {
        checkCudaErr(cudaMalloc(&data, size * sizeof(float)));
    }

    CudaTensor(const vector<int> &shape, float *ptr)
        : Tensor(shape, ptr, true) {
    }

    ~CudaTensor() {
        if (data)
            checkCudaErr(cudaFree(data));
        data = nullptr;
    }

    void fromHost(const void *ptr, size_t n_size) {
        CHECK(n_size == size);
        checkCudaErr(cudaMemcpy(
            data, ptr, n_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void fromHost(const std::vector<float> &vec) {
        CHECK(vec.size() == size);
        checkCudaErr(cudaMemcpy(data,
                                (void *)vec.data(),
                                size * sizeof(float),
                                cudaMemcpyHostToDevice));
    }

    void toHost(void *ptr, size_t n_size) {
        CHECK(n_size == size);
        checkCudaErr(cudaMemcpy(
            ptr, data, n_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void toHost(std::vector<float> &vec) {
        CHECK(vec.size() == size);
        checkCudaErr(cudaMemcpy((void *)vec.data(),
                                data,
                                size * sizeof(float),
                                cudaMemcpyDeviceToHost));
    }

    std::vector<float> toHost() {
        vector<float> vec(size);
        checkCudaErr(cudaMemcpy((void *)vec.data(),
                                data,
                                size * sizeof(float),
                                cudaMemcpyDeviceToHost));
        return vec;
    }

    void constant(const float value);

    void uniform();

    void normal(const float mean, const float stddev);
};