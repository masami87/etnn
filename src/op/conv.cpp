#include "conv.h"

Conv::Conv(cudnnTensorDescriptor_t in_desc,         // in_desc
           cudnnFilterDescriptor_t filt_desc,       // filt_desc
           cudnnTensorDescriptor_t bias_desc,       // bias_desc
           cudnnTensorDescriptor_t out_desc,        // out_desc
           cudnnConvolutionDescriptor_t conv_desc,  // conv_desc
           std::shared_ptr<CudaContext> &ctx_)
    : Op("Convolution", ctx_)
    , in_desc(in_desc)
    , filt_desc(filt_desc)
    , out_desc(out_desc)
    , conv_desc(conv_desc)
    , bias_desc(bias_desc) {
    // algorithm
    checkCudnnErr(cudnnGetConvolutionForwardAlgorithm(
        ctx->m_cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));
    bdata_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    bfilt_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    // workspace
    checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(ctx->m_cudnn,
                                                          in_desc,
                                                          filt_desc,
                                                          conv_desc,
                                                          out_desc,
                                                          algo,
                                                          &workspace_size));

    checkCudaErr(cudaMalloc(&workspace, workspace_size));
}

Conv::~Conv() {
    checkCudnnErr(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCudnnErr(cudnnDestroyFilterDescriptor(filt_desc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(in_desc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(out_desc));
    checkCudnnErr(cudnnDestroyTensorDescriptor(bias_desc));
    checkCudaErr(cudaFree(workspace));
}

// perform
void Conv::gpuForward(float *in_data,
                      float *filt_data,
                      float *bias_data,
                      float *out_data) {
    float alpha = 1.f;
    float beta  = 0.f;
    // conv
    checkCudnnErr(cudnnConvolutionForward(ctx->m_cudnn,
                                          &alpha,
                                          in_desc,
                                          in_data,
                                          filt_desc,
                                          filt_data,
                                          conv_desc,
                                          algo,
                                          workspace,
                                          workspace_size,
                                          &beta,
                                          out_desc,
                                          out_data));
    // bias
    checkCudnnErr(cudnnAddTensor(
        ctx->m_cudnn, &alpha, bias_desc, bias_data, &alpha, out_desc, out_data));
}

void Conv::gpuBackward(float *in_diff,
                       float *in_data,
                       float *filt_data,
                       float *filt_grad,
                       float *bias_data,
                       float *bias_grad,
                       float *diff) {
    float alpha = 1.f;
    float beta  = 0.f;
    checkCudnnErr(cudnnConvolutionBackwardFilter(ctx->m_cudnn,
                                                 &alpha,
                                                 in_desc,
                                                 in_data,
                                                 out_desc,
                                                 in_diff,
                                                 conv_desc,
                                                 bfilt_algo,
                                                 workspace,
                                                 workspace_size,
                                                 &beta,
                                                 filt_desc,
                                                 filt_grad));
    checkCudnnErr(cudnnConvolutionBackwardBias(
        ctx->m_cudnn, &alpha, out_desc, in_diff, &beta, bias_desc, bias_grad));
    checkCudnnErr(cudnnConvolutionBackwardData(ctx->m_cudnn,
                                               &alpha,
                                               filt_desc,
                                               filt_data,
                                               out_desc,
                                               in_diff,
                                               conv_desc,
                                               bdata_algo,
                                               workspace,
                                               workspace_size,
                                               &beta,
                                               in_desc,
                                               diff));
}
