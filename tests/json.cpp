#include <etnn/etnn.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv) {
    json j;
    InitParser(j, "../../json/mnist.json");
    std::cout << std::setw(4) << j << std::endl;

    auto ctx = createCudaContext();
    auto net = new Net(ctx);

    unordered_map<string, double> args;

    Load(j, net, args);

    net->print_layer();
    cout << "digits args:" << endl;
    for (auto a : args) {
        cout << a.first << "->" << a.second << endl;
    }

    delete net;
    return 0;
}