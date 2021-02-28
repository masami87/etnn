#pragma once

#include "data_input.h"
#include "layer.h"
#include "../network.h"

class Input : public Layer {
 public:
    Input(DataInput *parent, std::shared_ptr<CudaContext> ctx_ = nullptr);

    void Init() override;
    void Forward() override;
    void Backward() override;

 protected:
    DataInput *parent;
};