#include "utils.h"

#include <algorithm>
#include <iostream>
#include <random>  // std::default_random_engine

void print2D(float* ptr, int r, int c) {
    for (int i = 0; i < r; i++) {
        std::cout << "[ ";
        for (int j = 0; j < c; j++) {
            std::cout << *(ptr + i * c + j) << " ";
        }
        std::cout << "]\n";
    }
}

void printData(std::shared_ptr<FloatTensor> tensor) {
    size_t n   = tensor->size;
    float* ptr = new float[n];
    tensor->toHost(ptr, n);
    if (tensor->dim == 2 || (tensor->shape[2] == tensor->shape[3] == 1)) {
        print2D(ptr, tensor->shape[0], tensor->shape[1]);
        return;
    }
    std::cout << "[ ";
    for (size_t i = 0; i < n; i++) {
        std::cout << *(ptr + i) << " ";
    }
    std::cout << "]\n";
    delete[] ptr;
}

BatchDataset::BatchDataset(Datasets* dset, int b, bool sf, int seed)
    : dataset(dset)
    , batch(b)
    , shuffle(sf)
    , seed(seed) {
    dataset->load();
    CHECK(dataset->len != 0) << "The dataset's num is 0!\n";
    this->total_num       = dataset->len;
    this->total_data      = dataset->dataset_data;
    this->batch_num       = total_num / batch;
    this->num_feature     = dataset->num_feature;
    this->num_label       = dataset->num_label;
    this->data_pair.first = std::make_shared<FloatTensor>(
        std::initializer_list<int>{batch, num_feature});
    this->data_stride      = batch * num_feature;
    this->data_pair.second = std::make_shared<FloatTensor>(
        std::initializer_list<int>{batch, num_label});
    this->label_stride = batch * num_label;
    this->idx          = 0;
    this->batch_index  = vector<int>(batch_num);
    for (int i = 0; i < batch_num; i++) {
        this->batch_index[i] = i;
    }
    if (shuffle) {
        std::shuffle(batch_index.begin(),
                     batch_index.end(),
                     std::default_random_engine(seed));
    }
}

BatchDataset::~BatchDataset() {
    if (dataset)
        delete dataset;
    dataset = nullptr;
}

std::pair<std::shared_ptr<FloatTensor>, std::shared_ptr<FloatTensor>>& BatchDataset::
    fetch() {
    CHECK(idx < batch_num) << "There is no more data in the "
                              "BatchDataset!\n";
    int index = batch_index[idx];
    this->data_pair.first->fromHost(dataset->dataset_data + index * data_stride,
                                    data_stride);
    this->data_pair.second->fromHost(
        dataset->dataset_label + index * label_stride, label_stride);
    ++idx;
    return this->data_pair;
}

void BatchDataset::reset(int i) {
    this->idx = i;
}