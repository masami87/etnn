#include "../src/dataset.h"

#include <iostream>
#include <vector>

#include "../src/device.h"
#include "../src/layer/activation.h"
#include "../src/layer/conv2D.h"
#include "../src/layer/data_input.h"
#include "../src/layer/dense.h"
#include "../src/layer/input.h"
#include "../src/layer/loss_layer.h"
#include "../src/layer/softmax.h"
#include "../src/loss/cross_entropy.h"
#include "../src/network.h"
#include "../src/utils.h"

using namespace std;

inline void print(const vector<int>& v) {
    cout << "{";
    for (auto i : v) {
        cout << i << " ";
    }
    cout << "}\n";
}

int main() {
    int batch = 64;
    auto dataloader =
        new BatchDataset(new MNIST("/home/wt/aisys/etnn/data", false), batch);

    auto dp = dataloader->fetch();

    auto input_data = dp.first;
    auto label      = dp.second;
    print(input_data->shape);
    printData(label);

    return 0;
}