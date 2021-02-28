#pragma once

#include "../base.h"
#include "../tensor.h"
#include "op.h"

struct Conv final : public Op {
  Conv() = delete;
  Conv(cudnnTensorDescriptor_t in_desc,        // in_desc
       cudnnFilterDescriptor_t filt_desc,      // filt_desc
       cudnnTensorDescriptor_t bias_desc,      // bias_desc
       cudnnTensorDescriptor_t out_desc,       // out_desc
       cudnnConvolutionDescriptor_t conv_desc, // conv_desc
       std::shared_ptr<CudaContext> &ctx_);

  ~Conv() override;

  void gpuForward(float *in_data, float *filt_data, float *bias_data,
                  float *out_data);

  void gpuBackward(float *in_diff, float *in_data, float *filt_data,
                   float *filt_grad, float *bias_data, float *bias_grad,
                   float *diff);

  cudnnTensorDescriptor_t in_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t out_desc;

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionBwdDataAlgo_t bdata_algo;

  cudnnConvolutionBwdFilterAlgo_t bfilt_algo;
  size_t workspace_size; // extra size for computing
  void *workspace;       // pointer to the extra size
};