#pragma once

#include "../loss/loss.h"
#include "../network.h"
#include "./initializer/initializer.h"
#include "layer.h"

class LossLayer : public Layer {
 public:
    Loss* loss;

    LossLayer() = delete;

    LossLayer(const std::string& loss,
              std::shared_ptr<CudaContext> ctx_ = nullptr);

    ~LossLayer() = default;

    float loss_value(std::shared_ptr<FloatTensor> predict,
                     std::shared_ptr<FloatTensor> label);

    void loss_delta(std::shared_ptr<FloatTensor> predict,
                    std::shared_ptr<FloatTensor> label);

    void Init() override;

    void Forward() override;

    void Backward() override;
};