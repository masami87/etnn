#pragma once

#include <string>

#include "tensor.h"

class Datasets {
 public:
    float* dataset_data;
    float* dataset_label;
    std::string name;

    int len = 0;

    int num_feature = 0;

    int num_label = 0;

    bool is_test = false;

    Datasets(const std::string, bool is_test);

    virtual ~Datasets();

    virtual void load() = 0;
};

class MNIST : public Datasets {
 public:
    std::string directory;

    MNIST(const std::string& dir, bool is_test);

    void load() override;

 private:
    float* load_data(const std::string& filepath);

    float* load_label(const std::string& filepath);
};