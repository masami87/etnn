#pragma once

#include <string>
#include <vector>

#include "../device.h"
#include "../tensor.h"

class Loss {
 public:
    std::string name;

    std::shared_ptr<FloatTensor> gt;

    Loss(const std::string& name);

    ~Loss() = default;


    virtual float compute(std::shared_ptr<FloatTensor> output,
                          std::shared_ptr<FloatTensor> label) = 0;

    virtual void delta(std::shared_ptr<FloatTensor> output,
                       std::shared_ptr<FloatTensor> label,
                       std::shared_ptr<FloatTensor> diff) = 0;
};
