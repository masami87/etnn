#include <iostream>

#include "../src/initializer/initializer.h"
#include "../src/tensor.h"

using namespace std;
void print2D(float* ptr, int r, int c) {
    for (int i = 0; i < r; i++) {
        std::cout << "[ ";
        for (int j = 0; j < c; j++) {
            std::cout << *(ptr + i * c + j) << " ";
        }
        std::cout << "]\n";
    }
}

int main() {
    auto tensor  = new FloatTensor({10, 10});
    auto iconst  = IConstant(0.96);
    auto inormal = IRandNormal(3.9, 0.1);
    auto iunifor = IRandUniform();
    // cout << tensor->size << endl;
    // iconst.apply(tensor);
    inormal.apply(tensor);
    // iunifor.apply(tensor);
    float* data = new float[tensor->size];
    checkCudaErr(cudaMemcpy(data,
                            tensor->data_ptr(),
                            (tensor->size) * sizeof(float),
                            cudaMemcpyDeviceToHost));
    print2D(data, tensor->shape[0], tensor->shape[1]);
    cout << endl;
    // float data_ref[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    // print2D(data_ref, tensor->shape[0], tensor->shape[1]);

    return 0;
}