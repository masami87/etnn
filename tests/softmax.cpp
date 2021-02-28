#include "../src/layer/softmax.h"

#include <iostream>
#include <vector>

#include "../src/device.h"
#include "../src/layer/activation.h"
#include "../src/layer/conv2D.h"
#include "../src/layer/data_input.h"
#include "../src/layer/dense.h"
#include "../src/layer/input.h"
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

    std::shared_ptr<FloatTensor> input_data(new FloatTensor({2, 4, 1, 1}));
    input_data->uniform();

    auto data_input = new DataInput(input_data->shape);
    print(input_data->shape);
    auto input_layer = std::shared_ptr<Layer>(new Input(data_input, ctx));

    auto dense = std::shared_ptr<Layer>(new Dense("dense", 5, ctx));

    auto act = std::shared_ptr<Layer>(new Activation("relu", 0, ctx));

    auto sm = std::shared_ptr<Layer>(new Softmax(ctx));

    input_layer->addChild(dense);
    dense->addParent(input_layer);
    dense->addChild(act);
    act->addParent(dense);
    act->addChild(sm);
    sm->addParent(act);
    input_layer->Init();
    dense->Init();
    act->Init();
    sm->Init();
    data_input->FeedData(input_data);
    input_layer->Forward();
    dense->Forward();
    act->Forward();
    sm->Forward();

    cout << "dense output:" << endl;
    auto output = dense->get_output();
    print(output->shape);
    printData(output);

    cout << "act output:" << endl;
    auto act_output = act->get_output();
    print(act_output->shape);
    printData(act_output);

    cout << "sm output:" << endl;
    auto sm_output = sm->get_output();
    print(sm_output->shape);
    printData(sm_output);

    return 0;
}