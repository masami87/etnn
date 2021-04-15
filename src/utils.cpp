#include "utils.h"

#include <iostream>

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

BatchDataset::BatchDataset(Datasets* dset, int b)
    : dataset(dset)
    , batch(b) {
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
    this->batch_i      = 0;
}

BatchDataset::~BatchDataset() {
    if (dataset)
        delete dataset;
    dataset = nullptr;
}

std::pair<std::shared_ptr<FloatTensor>, std::shared_ptr<FloatTensor>>& BatchDataset::
    fetch() {
    CHECK(batch_i < batch_num) << "There is no more data in the "
                                  "BatchDataset!\n";
    this->data_pair.first->fromHost(
        dataset->dataset_data + batch_i * data_stride, data_stride);
    this->data_pair.second->fromHost(
        dataset->dataset_label + batch_i * label_stride, label_stride);
    ++batch_i;
    return this->data_pair;
}

void BatchDataset::reset(int i) {
    this->batch_i = i;
}