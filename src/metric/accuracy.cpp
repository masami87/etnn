#include "metric.h"

CategoricalAccuracy::CategoricalAccuracy()
    : Metric("CategoricalAccuracy") {
}

float CategoricalAccuracy::value(std::shared_ptr<FloatTensor> output,
                                 std::shared_ptr<FloatTensor> label) {
    auto l_shape = label->shape;
    if (l_shape.size() == 2) {
        l_shape = vector<int>{l_shape[0], l_shape[1], 1, 1};
    }
    CHECK(output->shape == l_shape);
    int batch     = output->shape[0];
    int feature   = output->shape[1];
    int sz        = output->size;
    auto h_output = new float[sz]();
    auto h_label  = new float[sz]();

    output->toHost(h_output, sz);
    label->toHost(h_label, sz);

    float acc_num = 0;

    for (int i = 0; i < batch; i++) {
        int predict = -1, gt = -1;
        float max_o = -1.f, max_l = -1.f;
        int offset = i * feature;
        for (int j = 0; j < feature; j++) {
            if (h_output[offset + j] > max_o) {
                max_o   = h_output[offset + j];
                predict = j;
            }
            if (h_label[offset + j] > max_o) {
                max_l = h_label[offset + j];
                gt    = j;
            }
        }
        acc_num += (predict == gt);
    }

    float acc = acc_num / batch;

    return acc;
}
