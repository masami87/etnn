#include "softmax.h"


Softmax::Softmax(std::shared_ptr<CudaContext> ctx_)
    : Layer("softmax", ctx_) {
    checkCudnnErr(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnnErr(cudnnCreateTensorDescriptor(&y_desc));
}

Softmax::~Softmax() {
    checkCudnnErr(cudnnDestroyTensorDescriptor(x_desc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(y_desc));
}

void Softmax::Init() {
    input       = parents[0]->get_output();
    input_shape = parents[0]->output_shape;
    CHECK(input_shape.size() == 4);
    output_shape = input_shape;

    checkCudnnErr(cudnnSetTensor4dDescriptor(x_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             input_shape[0],
                                             input_shape[1],
                                             input_shape[2],
                                             input_shape[3]));

    checkCudnnErr(cudnnSetTensor4dDescriptor(y_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             output_shape[0],
                                             output_shape[1],
                                             output_shape[2],
                                             output_shape[3]));

    output = std::make_shared<FloatTensor>(output_shape);
    diff   = std::make_shared<FloatTensor>(input_shape);
}

void Softmax::Forward() {
    float alpha = 1.0f;
    float beta  = 0.f;
    checkCudnnErr(cudnnSoftmaxForward(ctx->m_cudnn,
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      x_desc,
                                      input->data_ptr(),
                                      &beta,
                                      y_desc,
                                      output->data_ptr()));
}

void Softmax::Backward() {
    float alpha  = 1.0f;
    float beta   = 0.f;
    auto in_diff = childs[0]->get_diff();
    // 梯度是对的 == (a - y) a->output y->label
    checkCudnnErr(cudnnSoftmaxBackward(ctx->m_cudnn,
                                       CUDNN_SOFTMAX_ACCURATE,
                                       CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha,
                                       y_desc,
                                       output->data_ptr(),
                                       y_desc,
                                       in_diff->data_ptr(),
                                       &beta,
                                       x_desc,
                                       diff->data_ptr()));
}

void Softmax::updateWeights(float lr) {
    /* do nothing */
}