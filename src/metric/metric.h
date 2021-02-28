#pragma once

#include "../tensor.h"

class Metric {
 public:
    std::string name;

    Metric() = delete;

    Metric(const std::string&);

    ~Metric() = default;

    virtual float value(std::shared_ptr<FloatTensor> predict,
                        std::shared_ptr<FloatTensor> Tensor) = 0;
};

class CategoricalAccuracy : public Metric {
 public:
    CategoricalAccuracy();

    ~CategoricalAccuracy() = default;

    float value(std::shared_ptr<FloatTensor> predict,
                std::shared_ptr<FloatTensor> Tensor) override;
};