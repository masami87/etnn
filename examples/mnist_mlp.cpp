#include <iostream>
#include <vector>

#include "../src/dataset.h"
#include "../src/device.h"
#include "../src/layer/activation.h"
#include "../src/layer/conv2D.h"
#include "../src/layer/data_input.h"
#include "../src/layer/dense.h"
#include "../src/layer/input.h"
#include "../src/layer/loss_layer.h"
#include "../src/layer/softmax.h"
#include "../src/loss/cross_entropy.h"
#include "../src/metric/metric.h"
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
    auto ctx = createCudaContext();

    int batch       = 64;
    auto dataloader = new BatchDataset(new MNIST("../../data", false), batch);

    auto testloader = new BatchDataset(new MNIST("../../data", true), batch);

    auto dp = dataloader->fetch();

    auto input_data = dp.first;
    auto label      = dp.second;
    input_data->Reshape({batch, 28 * 28});

    // printData(input_data);

    auto metric = new CategoricalAccuracy();

    auto model = new Net(ctx);
    model->set_data(input_data->shape);

    model->add_layer(std::make_shared<Dense>("fc2", 512));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc2", 64));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc3", 10));
    model->add_layer(std::make_shared<Softmax>());

    model->compile("cross_entropy", std::make_shared<CategoricalAccuracy>());

    cout << model->params() << endl;

    model->print_layer();

    model->fit(100, 0.005, dataloader, testloader);

    return 0;
}