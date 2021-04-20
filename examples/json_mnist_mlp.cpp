#include <etnn/etnn.h>

#include <iomanip>
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
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << "<the path to json file> <the path to mnist dataset>"
                  << endl;
        return 1;
    }
    std::string json_path = argv[1];
    std::string data_path = argv[2];
    json j;
    InitParser(j, std::move(json_path));

    // std::cout << std::setw(4) << j << std::endl;

    auto ctx = createCudaContext();
    auto net = new Net(ctx);
    unordered_map<string, double> args;

    // load network and args from json
    Load(j, net, args);

    int batch = static_cast<int>(args["batch"]);
    float lr  = static_cast<float>(args["lr"]);
    int epoch = static_cast<int>(args["epoch"]);

    auto dataloader =
        new BatchDataset(new MNIST(data_path, false), batch, true);

    auto testloader = new BatchDataset(new MNIST(data_path, true), batch);

    net->compile("cross_entropy", std::make_shared<CategoricalAccuracy>());

    cout << "The param size of model is: " << net->params() << endl;

    net->print_layer();

    net->fit(epoch, lr, dataloader, testloader);

    delete net;

    return 0;
}