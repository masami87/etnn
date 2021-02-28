#pragma once

#include <memory>

#include "../op/conv.h"
#include "layer.h"

class Conv2D final : public Layer {
 public:
    int in_channel;
    int out_channel;
    int kernel;
    int stride;
    int pad;
    explicit Conv2D(const std::string &name,
                    int in_channel,
                    int out_channel,
                    int kernel,
                    int stride                        = 1,
                    int pad                           = 1,
                    std::shared_ptr<CudaContext> ctx_ = nullptr);

    Conv2D(const Conv2D &) = delete;
    Conv2D &operator=(Conv2D &) = delete;

    ~Conv2D();

    void Init() override;
    void Forward() override;

    void Backward() override;

    void updateWeights(float lr) override;

 private:
    std::shared_ptr<Conv> convop;
};