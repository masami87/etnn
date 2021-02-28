#include "input.h"

#include <vector>

Input::Input(DataInput *parent, std::shared_ptr<CudaContext> ctx_)
    : Layer("input", ctx_)
    , parent(parent) {
    input_shape  = parent->output_shape;
    output_shape = parent->output_shape;
}

void Input::Init() {
}

void Input::Forward() {
    output = parent->getData();
    CHECK(output != nullptr);
}
void Input::Backward() {
}
