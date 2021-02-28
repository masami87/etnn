#include "conv2D.h"

Conv2D::Conv2D(const std::string &name,
               int in_channel,
               int out_channel,
               int kernel,
               int stride,
               int pad,
               std::shared_ptr<CudaContext> ctx_)
    : Layer(name, ctx_)
    , in_channel(in_channel)
    , out_channel(out_channel)
    , kernel(kernel)
    , stride(stride)
    , pad(pad) {
}

Conv2D::~Conv2D() {
}

void Conv2D::Init() {
    // Init filter desc
    cudnnFilterDescriptor_t filt_desc;
    cudnnTensorDescriptor_t bias_desc;
    checkCudnnErr(cudnnCreateFilterDescriptor(&filt_desc));
    checkCudnnErr(cudnnCreateTensorDescriptor(&bias_desc));

    checkCudnnErr(cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channel, 1, 1));
    checkCudnnErr(cudnnSetFilter4dDescriptor(filt_desc,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW,
                                             out_channel,
                                             in_channel,
                                             kernel,
                                             kernel));

    auto filter_size = {out_channel, in_channel, kernel, kernel};
    auto bias_size   = {1, out_channel, 1, 1};

    params.clear();
    // params[0] -> filter , params[1] -> bias
    params.push_back(std::make_shared<FloatTensor>(filter_size));
    params.push_back(std::make_shared<FloatTensor>(bias_size));

    // initialize
    params[0]->normal(0.0, 1.0);
    params[1]->normal(0.0, 1.0);

    gradients.push_back(std::make_shared<FloatTensor>(filter_size));
    gradients.push_back(std::make_shared<FloatTensor>(bias_size));

    // Init input desc
    // input       = parents[0]->get_output();
    input_shape = parents[0]->output_shape;
    CHECK(input_shape.size() == 4);
    CHECK(input_shape[1] == in_channel);
    cudnnTensorDescriptor_t in_desc;
    checkCudnnErr(cudnnCreateTensorDescriptor(&in_desc));
    checkCudnnErr(cudnnSetTensor4dDescriptor(in_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             input_shape[0],
                                             input_shape[1],
                                             input_shape[2],
                                             input_shape[3]));
    diff = std::make_shared<FloatTensor>(input_shape);

    // Init Conv desc
    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnnErr(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCudnnErr(cudnnSetConvolution2dDescriptor(conv_desc,
                                                  pad,
                                                  pad,
                                                  stride,
                                                  stride,
                                                  1,
                                                  1,
                                                  CUDNN_CONVOLUTION,
                                                  CUDNN_DATA_FLOAT));

    // Init output desc
    int out_n;
    int out_c;
    int out_h;
    int out_w;
    checkCudnnErr(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));

    output_shape = {out_n, out_c, out_h, out_w};

    cudnnTensorDescriptor_t out_desc;
    checkCudnnErr(cudnnCreateTensorDescriptor(&out_desc));
    checkCudnnErr(cudnnSetTensor4dDescriptor(out_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             out_n,
                                             out_c,
                                             out_h,
                                             out_w));

    output = std::make_shared<FloatTensor>(output_shape);

    // create Conv Operator
    convop = std::make_shared<Conv>(
        in_desc, filt_desc, bias_desc, out_desc, conv_desc, ctx);
}

void Conv2D::Forward() {
    input           = parents[0]->get_output();
    auto input_ptr  = input->data_ptr();
    auto filter_ptr = params[0]->data_ptr();
    auto bias_ptr   = params[1]->data_ptr();
    auto output_ptr = output->data_ptr();
    convop->gpuForward(input_ptr, filter_ptr, bias_ptr, output_ptr);
}

void Conv2D::Backward() {
    auto in_diff     = childs[0]->get_diff()->data_ptr();
    auto input_ptr   = input->data_ptr();
    auto filter_ptr  = params[0]->data_ptr();
    auto filter_grad = gradients[0]->data_ptr();
    auto bias_ptr    = params[1]->data_ptr();
    auto bias_grad   = gradients[1]->data_ptr();
    auto diff_ptr    = diff->data_ptr();
    convop->gpuBackward(in_diff,
                        input_ptr,
                        filter_ptr,
                        filter_grad,
                        bias_ptr,
                        bias_grad,
                        diff_ptr);
}

void Conv2D::updateWeights(float lr) {
    float alpha = -1.f * lr;
    // filter
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