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

    int batch = 64;
    auto dataloader =
        new BatchDataset(new MNIST("/home/wt/aisys/etnn/data", false), batch);

    auto testloader =
        new BatchDataset(new MNIST("/home/wt/aisys/etnn/data", true), batch);

    auto dp = dataloader->fetch();

    auto input_data = dp.first;
    auto label      = dp.second;
    input_data->Reshape({batch, 1, 28, 28});

    // printData(input_data);

    auto metric = new CategoricalAccuracy();

    auto model = new Net(ctx);
    model->set_data(input_data->shape);

    model->add_layer(std::make_shared<Conv2D>("conv1", 1, 16, 3, 1, 1));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    // model->add_layer(std::make_shared<Conv2D>("conv2", 16, 32, 3, 1, 1));
    // model->add_layer(std::make_shared<Activation>("sigmoid"));
    // model->add_layer(std::make_shared<Conv2D>("conv3", 32, 64, 3, 1));
    // model->add_layer(std::make_shared<Activation>("sigmoid"));
    // model->add_layer(std::make_shared<Dense>("fc1", 256));
    // TODO:
    // 通过观察relu层的输出和sigmoid的输出对比发现应该是relu层的输出权值太大导致训练crash了
    // TODO: 明天记录一下这个问题，改成clipped_relu之后就又可以训练了
    // model->add_layer(std::make_shared<Activation>("clipped_relu", 1));
    model->add_layer(std::make_shared<Dense>("fc2", 512));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc2", 64));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc3", 10));
    // model->add_layer(std::make_shared<Activation>("relu"));
    model->add_layer(std::make_shared<Softmax>());

    model->compile("cross_entropy", std::make_shared<CategoricalAccuracy>());

    cout << model->params() << endl;

    model->print_layer();

    model->fit(100, 0.005, dataloader, testloader);

    return 0;
}