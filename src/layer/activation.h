#pragma once

#include "layer.h"

class Activation : public Layer {
 public:
    enum class Type : int {
        RELU,
        SIGMOID,
        ELU,
        TANH,
    };
    cudnnActivationMode_t mode;
    
    /**
     * @brief Floating point number. When the activation mode (see
    cudnnActivationMode_t) is set to CUDNN_ACTIVATION_CLIPPED_RELU, this input
    specifies the clipping threshold; and when the activation mode is set to
    CUDNN_ACTIVATION_RELU, this input specifies the upper bound.
     *
     */
    double coef;

    cudnnActivationDescriptor_t activationDesc;

    cudnnTensorDescriptor_t input_desc;

    cudnnTensorDescriptor_t output_desc;

    Activation(const std::string& name,
                        double coef                      = 0,
                        std::shared_ptr<CudaContext> ctx = nullptr);

    ~Activation();

    void Init() override;

    void Forward() override;

    void Backward() override;

    void updateWeights(float lr) override;
};