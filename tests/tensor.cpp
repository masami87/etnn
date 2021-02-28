#include "tensor.h"

#include <iostream>
#include <vector>

using namespace std;

void print(const vector<int> &v) {
    cout << "{";
    for (auto i : v) {
        cout << i << " ";
    }
    cout << "}\n";
}

int main() {
    float a  = 7.9;
    float *p = new float[2000];
    Tensor t({23, 42, 12}, p);
    print(t.shape);
    print(t.stride);
    cout << t.dim << " " << t.size << endl;

    Tensor tt = std::move(t);
    cout << t.data_ptr() << endl;
    cout << tt.data_ptr() << endl;

    tt.value({2, 3, 4}) = 3.3;
    cout << tt.value({2, 3, 4}) << endl;

    std::shared_ptr<FloatTensor> xx(new FloatTensor({1, 3, 4}));

    FloatTensor *cut = new FloatTensor({3, 2, 4});
    //   auto spp = make_shared<FloatTensor>({1, 2, 3});
    print(cut->shape);
    return 0;
}