#include "activation.h"

#include <unordered_map>

static std::unordered_map<std::string, cudnnActivationMode_t> ActivationMap = {
    {"relu", CUDNN_ACTIVATION_RELU},
    {"clipped_relu", CUDNN_ACTIVATION_CLIPPED_RELU},
    {"elu", CUDNN_ACTIVATION_ELU},
    {"sigmoid", CUDNN_ACTIVATION_SIGMOID},
    {"tanh", CUDNN_ACTIVATION_TANH},
};

Activation::Activation(const std::string& name,
                       double coef,
                       std::shared_ptr<CudaContext> ctx)
    : Layer(name, ctx)
    , coef(coef) {
    CHECK(ActivationMap.find(name) != ActivationMap.end())
        << "There is no activation : " << name << std::endl;
    this->mode = ActivationMap[name];
}

Activation::~Activation() {
    checkCudnnErr(cudnnDestroyActivationDescriptor(activationDesc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(input_desc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(output_desc));
}

void Activation::Init() {
    checkCudnnErr(cudnnCreateActivationDescriptor(&activationDesc));
    checkCudnnErr(cudnnSetActivationDescriptor(
        activationDesc, mode, CUDNN_PROPAGATE_NAN, coef));
    CHECK(parents.size() == 1);
    input_shape = parents[0]->output_shape;
    CHECK(input_shape.size() == 4);
    output_shape = input_shape;
    input        = parents[0]->get_output();
    checkCudnnErr(cudnnCreateTensorDescriptor(&input_desc));
    checkCudnnErr(cudnnSetTensor4dDescriptor(input_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             input_shape[0],
                                             input_shape[1],
                                             input_shape[2],
                                             input_shape[3]));

    checkCudnnErr(cudnnCreateTensorDescriptor(&output_desc));
    checkCudnnErr(cudnnSetTensor4dDescriptor(output_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             output_shape[0],
                                             output_shape[1],
                                             output_shape[2],
                                             output_shape[3]));
    output = std::make_shared<FloatTensor>(output_shape);
    diff   = std::make_shared<FloatTensor>(input_shape);
}

void Activation::Forward() {
    float alpha = 1.0f;
    float beta  = 0.f;
    checkCudnnErr(cudnnActivationForward(ctx->m_cudnn,
                                         activationDesc,
                                         &alpha,
                                         input_desc,
                                         input->data_ptr(),
                                         &beta,
                                         output_desc,
                                         output->data_ptr()));
}

void Activation::Backward() {
    float alpha  = 1.0f;
    float beta   = 0.f;
    auto in_diff = childs[0]->get_diff();
    checkCudnnErr(cudnnActivationBackward(ctx->m_cudnn,
                                          activationDesc,
                                          &alpha,
                                          output_desc,
                                          output->data_ptr(),
                                          output_desc,
                                          in_diff->data_ptr(),
                                          input_desc,
                                          input->data_ptr(),
                                          &beta,
                                          input_desc,
                                          diff->data_ptr()));
}

void Activation::updateWeights(float lr) {
    /* do nothing */
}