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

void StartServer() {
    if (!ps::IsServer())
        return;
    std::cout << "Server rank: " << ps::MyRank() << " is running." << std::endl;
    auto server = new SgdDistServer(0.01, 1);  // 同步
    ps::RegisterExitCallback([server]() { delete server; });
}

void RunWorker(std::string& file_path) {
    if (!ps::IsWorker()) {
        return;
    }

    int rank       = ps::MyRank();
    int num_wokers = ps::NumWorkers();
    std::cout << "Worker rank: [" << rank + 1 << "]/[" << num_wokers << "] "
              << " is running." << std::endl;

    auto ctx = createCudaContext();

    int batch = 64;
    auto dataloader =
        new BatchDataset(new MNIST(file_path, false, rank, num_wokers), batch);

    auto testloader =
        new BatchDataset(new MNIST(file_path, true, rank, num_wokers), batch);

    auto dp = dataloader->fetch();

    auto input_data = dp.first;
    auto label      = dp.second;
    input_data->Reshape({batch, 28 * 28});

    // printData(input_data);

    auto worker = std::make_shared<DistWorker>();

    auto model = new NetDistributed(worker, ctx);
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
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << "<the path to mnist dataset>"
                  << endl;
        return 1;
    }
    std::string path = argv[1];

    ps::Start(0);

    StartServer();

    RunWorker(path);

    ps::Finalize(0, true);

    return 0;
}