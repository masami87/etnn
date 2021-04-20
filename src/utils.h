#pragma once
#include <stddef.h>

#include "dataset.h"
#include "device.h"
#include "tensor.h"

void print2D(float* ptr, int r, int c);
void printData(std::shared_ptr<FloatTensor> tensor);

class BatchDataset {
 public:
    Datasets* dataset;

    int batch;

    int total_num = 0;

    int batch_num = 0;

    int num_feature = 0;

    int num_label = 0;

    int data_stride = 0;

    int label_stride = 0;

    size_t idx = 0;

    vector<int> batch_index;

    unsigned long seed = 0;

    bool shuffle = false;

    BatchDataset(Datasets*, int batch, bool shuffle = false, int seed = 42);

    ~BatchDataset();

    std::pair<std::shared_ptr<FloatTensor>, std::shared_ptr<FloatTensor>>& fetch();

    void reset(int i = 0);

 private:
    std::pair<std::shared_ptr<FloatTensor>, std::shared_ptr<FloatTensor>>
        data_pair;

    float* total_data;

    float* total_label;
};
