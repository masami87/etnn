#include "tensor.h"

#include "initializer/initializer.h"

void CudaTensor::constant(const float value) {
    auto iconst = IConstant(value);
    iconst.apply(this);
}

void CudaTensor::uniform() {
    auto iuniform = IRandUniform();
    iuniform.apply(this);
}

void CudaTensor::normal(const float mean, const float stddev) {
    auto inormal = IRandNormal(mean, stddev);
    inormal.apply(this);
}