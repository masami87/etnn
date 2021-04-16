#include <etnn/etnn.h>

#include <iostream>
#include <vector>

using namespace std;

inline void print(const vector<int>& v) {
    cout << "{";
    for (auto i : v) {
        cout << i << " ";
    }
    cout << "}\n";
}
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << "<the path to mnist dataset>"
                  << endl;
        return 1;
    }
    std::string path = argv[1];

    auto ctx = createCudaContext();

    int batch       = 64;
    auto dataloader = new BatchDataset(new MNIST(path, false), batch, true);

    auto testloader = new BatchDataset(new MNIST(path, true), batch);

    auto dp = dataloader->fetch();

    auto input_data = dp.first;
    auto label      = dp.second;
    input_data->Reshape({batch, 28 * 28});

    // printData(input_data);

    auto model = new Net(ctx);
    model->set_data(input_data->shape);

    model->add_layer(std::make_shared<Dense>("fc1", 512));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc2", 64));
    model->add_layer(std::make_shared<Activation>("sigmoid"));
    model->add_layer(std::make_shared<Dense>("fc3", 10));
    model->add_layer(std::make_shared<Softmax>());

    model->compile("cross_entropy", std::make_shared<CategoricalAccuracy>());

    cout << "The param size of model is: " << model->params() << endl;

    model->print_layer();

    model->fit(100, 0.005, dataloader, testloader);

    return 0;
}