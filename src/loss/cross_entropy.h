#pragma once

#include "loss.h"

class CrossEntropy : public Loss {
 public:
    CrossEntropy();

    float compute(std::shared_ptr<FloatTensor> output,
                  std::shared_ptr<FloatTensor> label) override;

    void delta(std::shared_ptr<FloatTensor> output,
               std::shared_ptr<FloatTensor> label,
               std::shared_ptr<FloatTensor> diff) override;
};
