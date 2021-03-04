#pragma once

#include "../device.h"
#include "../tensor.h"

class Initializer {
 public:
    virtual void apply(FloatTensor *tensor) = 0;  // Pure virtual
};

class IConstant : public Initializer {
 public:
    float value;

    explicit IConstant(float v)
        : value(v) {
    }

    void apply(FloatTensor *tensor) override;
};

class IRandNormal : public Initializer {
 public:
    float mean;
    float stddev;

    IRandNormal(float mean, float stddev)
        : mean(mean)
        , stddev(stddev) {
    }

    void apply(FloatTensor *tensor) override;
};

class IRandUniform : public Initializer {
 public:
    IRandUniform() {
    }

    void apply(FloatTensor *tensor) override;
};