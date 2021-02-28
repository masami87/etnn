#include "loss_layer.h"

#include <unordered_map>

#include "../loss/cross_entropy.h"

LossLayer::LossLayer(const std::string& loss, std::shared_ptr<CudaContext> ctx_)
    : Layer(loss, ctx_) {
    if (loss == "cross_entropy") {
        this->loss = new CrossEntropy();
    } else {
        std::cout << "Unknow loss type!" << std::endl;
        abort();
    }
}

float LossLayer::loss_value(std::shared_ptr<FloatTensor> predict,
                            std::shared_ptr<FloatTensor> label) {
    return loss->compute(predict, label);
}

void LossLayer::loss_delta(std::shared_ptr<FloatTensor> predict,
                           std::shared_ptr<FloatTensor> label) {
    loss->delta(predict, label, this->diff);
}

void LossLayer::Init() {
    input_shape = parents[0]->output_shape;
    input       = parents[0]->get_output();
    diff        = std::make_shared<FloatTensor>(input_shape);
}

void LossLayer::Forward() {
}

void LossLayer::Backward() {
}