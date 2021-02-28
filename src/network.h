#pragma once

#include <stdio.h>

#include <string>
#include <vector>

#include "device.h"
#include "metric/metric.h"
#include "utils.h"

class Layer;

class DataInput;

class LossLayer;

class Net {
 public:
    size_t params_size = 0;

    float lr;

    Net(std::shared_ptr<CudaContext> ctx_);

    Net(std::shared_ptr<CudaContext>, vector<std::shared_ptr<Layer>>&);

    std::shared_ptr<FloatTensor> forward(
        const std::shared_ptr<FloatTensor> input);

    float compute_loss(std::shared_ptr<FloatTensor> predict,
                       std::shared_ptr<FloatTensor> label);

    void backward();

    void updateWeights();

    size_t params() const;

    void set_loss(const std::string& loss_name);

    void set_data(vector<int>& shape);

    void set_lr(const float learning_rate);

    void add_layer(std::shared_ptr<Layer>);

    void print_layer() const;

    void compile(const std::string& loss_name, std::shared_ptr<Metric> metric);

    void fit(const int epoch,
             const float learning_rate,
             BatchDataset* train_data,
             BatchDataset* test_data = nullptr);

 private:
    void Init();

    void InitCtx();

    void add(std::shared_ptr<Layer>);

    void train(int epoch, BatchDataset* train_data, BatchDataset* test_data);

    void test(BatchDataset* test_data);

    vector<std::shared_ptr<Layer>> layers;

    std::shared_ptr<CudaContext> ctx;

    std::shared_ptr<LossLayer> loss;

    std::shared_ptr<Metric> metric;

    std::shared_ptr<DataInput> data_input;
};