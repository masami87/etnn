#pragma once

#include "layer.h"

class Softmax : public Layer {
 public:
    cudnnTensorDescriptor_t x_desc;

    cudnnTensorDescriptor_t y_desc;

    Softmax(std::shared_ptr<CudaContext> ctx_ = nullptr);

    ~Softmax();

    void Init() override;

    void Forward() override;

    void Backward() override;

    void updateWeights(float lr) override;
};