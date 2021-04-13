#pragma once

#include <unordered_map>
#include <vector>

#include "base.h"
#include "ps/ps.h"

class DistWorker {
 private:
    ps::KVWorker<float>* worker_;

 public:
    DistWorker()
        : worker_(new ps::KVWorker<float>(0, 0)) {
    }

    ~DistWorker() {
        if (worker_)
            delete worker_;
        worker_ = nullptr;
    }

    void push(std::vector<ps::Key> keys,
              const std::vector<std::vector<float>>& gradients) {
        CHECK(keys.size() == gradients.size()) << "keys.size() must eaqul to "
                                                  "gradients.size()";
        size_t n = keys.size();
        std::vector<int> lens(n);
        size_t total = 0;
        for (int i = 0; i < n; i++) {
            lens[i] = gradients[i].size();
            total += lens[i];
        }

        size_t idx = 0;
        std::vector<float> values(total);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < lens[i]; j++) {
                values[idx++] = gradients[i][j];
            }
        }

        worker_->Wait(worker_->Push(keys, values, lens));
    }

    void pull(std::vector<ps::Key> keys,
              std::vector<std::vector<float>>& weights) {
        size_t n = keys.size();
        std::vector<int> lens(n);
        std::vector<float> values;

        worker_->Wait(worker_->Pull(keys, &values, &lens));

        CHECK(lens.size() == n);

        size_t idx = 0;
        for (int i = 0; i < n; i++) {
            CHECK(weights[i].size() == lens[i]);
            for (int j = 0; j < weights[i].size(); j++) {
                weights[i][j] = values[idx++];
            }
        }
    }
};