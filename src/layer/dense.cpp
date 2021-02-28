#include "dense.h"

#include "layer.h"
Dense::Dense(const std::string& name,
             int output_size,
             std::shared_ptr<CudaContext> ctx_)
    : Layer(name, ctx_)
    , output_size(output_size) {
}

Dense::~Dense() {
}

void Dense::Init() {
    this->input_shape = parents[0]->output_shape;
    input             = parents[0]->get_output();

    if (input_shape.size() == 2) {
        input_shape = {input_shape[0], input_shape[1], 1, 1};
    }
    CHECK(input_shape.size() == 4);
    // dim 4
    batch      = input_shape[0];
    input_size = input_shape[1] * input_shape[2] * input_shape[3];

    this->output_shape = vector<int>{batch, output_size, 1, 1};

    params.clear();
    params.push_back(std::make_shared<FloatTensor>(
        std::initializer_list<int>{output_size, input_size}));
    params.push_back(std::make_shared<FloatTensor>(
        std::initializer_list<int>{1, output_size}));
    params[0]->normal(0.0, 1.0);
    params[1]->normal(0.0, 1.0);

    gradients.clear();
    gradients.push_back(std::make_shared<FloatTensor>(
        std::initializer_list<int>{output_size, input_size}));
    gradients.push_back(std::make_shared<FloatTensor>(
        std::initializer_list<int>{1, output_size}));

    diff = std::make_shared<FloatTensor>(input_shape);

    output = std::make_shared<FloatTensor>(output_shape);

    one = std::make_shared<FloatTensor>(std::initializer_list<int>{batch, 1});
    one->constant(1.0f);
}

void Dense::Forward() {
    float alpha   = 1.0f;
    float beta    = 0.f;
    auto data     = parents[0]->get_output()->data_ptr();
    auto weight   = params[0]->data_ptr();
    auto bias     = params[1]->data_ptr();
    auto out      = output->data_ptr();
    auto one_data = one->data_ptr();
    
    // the memory layout of FloatTensor is NCHW, but cublas matrix's is column
    // major. For more information:https://www.freesion.com/article/5429183885/

    // -> means this read format leading to a auto trasnformation of dimention
    // data(batch, input) -> (input, batch)
    // weight(output, input) -> (input, output)  trans to (output, input)
    // out = (output, batch) -> (batch, output)

    checkCublasErr(cublasSgemm(ctx->m_cublas,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               output_size,
                               batch,
                               input_size,
                               &alpha,
                               weight,
                               input_size,
                               data,
                               input_size,
                               &beta,
                               out,
                               output_size));

    // out (batch ,output) -> (output, batch)
    // bias(1, output) -> (output, 1)
    // one(batch, 1) -> (1, batch)
    // out = out +  (output, 1) * (1, batch)

    checkCublasErr(cublasSgemm(ctx->m_cublas,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               output_size,
                               batch,
                               1,
                               &alpha,
                               bias,
                               output_size,
                               one_data,
                               1,
                               &alpha,
                               out,
                               output_size));
}

void Dense::Backward() {
    float alpha      = 1.0f;
    float beta       = 0.f;
    auto in_diff     = childs[0]->get_diff()->data_ptr();
    auto data        = parents[0]->get_output()->data_ptr();
    auto weight      = params[0]->data_ptr();
    auto weight_grad = gradients[0]->data_ptr();
    auto bias_grad   = gradients[1]->data_ptr();
    auto one_data    = one->data_ptr();
    auto diff_ptr    = diff->data_ptr();
    // https://zhuanlan.zhihu.com/p/47002393
    // weight (output, input)
    // y = wx + b    dw = x * dy
    // data(batch ,input) -> (input ,batch)
    // in_diff(batch, output) -> (output, batch)
    // w_grad = (input, batch) * (batch, output) = (input, output) -> (output,
    // input)
    checkCublasErr(cublasSgemm(ctx->m_cublas,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               input_size,
                               output_size,
                               batch,
                               &alpha,
                               data,
                               input_size,
                               in_diff,
                               output_size,
                               &beta,
                               weight_grad,
                               input_size));

    checkCublasErr(cublasSgemv(ctx->m_cublas,
                               CUBLAS_OP_N,
                               output_size,
                               batch,
                               &alpha,
                               in_diff,
                               output_size,
                               one_data,
                               1,
                               &beta,
                               bias_grad,
                               1));
    // dx = w * dy
    // diff(batch, input) -> (input, batch)
    // weight(output, input) -> (input, output)
    // in_diff (batch, output) -> (output, batch)
    checkCublasErr(cublasSgemm(ctx->m_cublas,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               input_size,
                               batch,
                               output_size,
                               &alpha,
                               weight,
                               input_size,
                               in_diff,
                               output_size,
                               &beta,
                               diff_ptr,
                               input_size));
}

void Dense::updateWeights(float lr) {
    float alpha = -1.f * lr;

    // weight
    checkCublasErr(cublasSaxpy_v2(ctx->m_cublas,
                                  params[0]->get_size(),
                                  &alpha,
                                  gradients[0]->data_ptr(),
                                  1,
                                  params[0]->data_ptr(),
                                  1));
    // bias
    checkCublasErr(cublasSaxpy_v2(ctx->m_cublas,
                                  params[1]->get_size(),
                                  &alpha,
                                  gradients[1]->data_ptr(),
                                  1,
                                  params[1]->data_ptr(),
                                  1));
}