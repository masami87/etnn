#pragma once

#include "layer.h"

class Dense : public Layer {
 public:
    int batch;
    int input_size;
    int output_size;
    std::shared_ptr<FloatTensor> one;
    Dense(const std::string &name,
                   int output_size,
                   std::shared_ptr<CudaContext> ctx = nullptr);
    ~Dense();

    void Init() override;

    void Forward() override;

    void Backward() override;

    void updateWeights(float lr) override;
};