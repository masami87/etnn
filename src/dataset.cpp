#include "dataset.h"

#include <fstream>
#include <iostream>

static int read_int(std::ifstream& f);

Datasets::Datasets(const std::string name, bool is_test)
    : name(name)
    , is_test(is_test) {
    dataset_data  = nullptr;
    dataset_label = nullptr;
}

Datasets::~Datasets() {
    if (dataset_data)
        delete[] dataset_data;
    if (dataset_label)
        delete[] dataset_label;
}

MNIST::MNIST(const std::string& dir, bool is_test)
    : Datasets("mnist", is_test)
    , directory(dir) {
}

float* MNIST::load_data(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "MNIST file not found!";

    // magic number
    int mn = read_int(ifs);

    int n;     // number of images
    int rows;  // number of rows
    int cols;  // number of columns
    n    = read_int(ifs);
    rows = read_int(ifs);
    cols = read_int(ifs);

    vector<int> shape{n, rows * cols};

    this->len         = shape[0];
    this->num_feature = shape[1];

    int size = 1;  // number of total elements
    for (auto& d : shape)
        size *= d;

    auto ptr = new float[size];
    for (int i = 0; i < size; i++) {
        uint8_t tmp;
        ifs.read((char*)&tmp, sizeof(tmp));
        ptr[i] = (float)tmp / 255;
    }
    return ptr;
}

float* MNIST::load_label(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "MNIST file not found!";
    // magic number
    int mn = read_int(ifs);
    // number of images
    int n = read_int(ifs);

    vector<int> shape{n, 10};

    CHECK(n == this->len);
    this->num_label = 10;

    auto ptr = new float[n * 10]();
    for (int i = 0; i < n; i++) {
        uint8_t tmp;
        ifs.read((char*)&tmp, sizeof(tmp));
        ptr[i * 10 + tmp] = 1.0f;
    }

    return ptr;
}

void MNIST::load() {
    if (!is_test) {
        dataset_data  = load_data(this->directory + "/train-images-idx3-ubyte");
        dataset_label = load_label(this->directory +
                                   "/train-labels-idx1-"
                                   "ubyte");
    } else {
        dataset_data  = load_data(this->directory + "/t10k-images-idx3-ubyte");
        dataset_label = load_label(this->directory + "/t10k-labels-idx1-ubyte");
    }
}

static int read_int(std::ifstream& f) {
    // 比如一个magic number是0x0083,在ifstream存储的顺序是0,0,8,3
    // 但是如果直接读到一个int*, 两个0会被读到低地址，即顺序是反的
    // 即如果使用 int* c; f.read((char*)&c, sizeof(int));
    // 那么c里面是0x3800
    int d = 0;
    int c;
    for (int i = 0; i < sizeof(int); i++) {
        c = 0;
        f.read((char*)&c, 1);
        d |= (c << (8 * (sizeof(int) - i - 1)));
    }
    return d;
}