#include "layer.h"

Layer::Layer(const std::string &name, std::shared_ptr<CudaContext> &ctx_)
    : name(name)
    , ctx(ctx_) {
}

Layer::~Layer() {
}

void Layer::addParent(const std::shared_ptr<Layer> &l) {
    this->parents.push_back(l);
}
void Layer::addChild(const std::shared_ptr<Layer> &l) {
    this->childs.push_back(l);
}
void Layer::updateWeights(float lr) {
}

std::shared_ptr<FloatTensor> Layer::get_output() const {
    return output;
}

std::shared_ptr<FloatTensor> Layer::get_diff() const {
    return diff;
}

size_t Layer::params_size() const {
    size_t sum = 0;
    for (auto &p : params) {
        sum += p->size;
    }
    return sum;
}
