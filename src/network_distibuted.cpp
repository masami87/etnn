#include "network_distibuted.h"

#include <unistd.h>

#include <iomanip>
#include <iostream>

#include "./layer/data_input.h"
#include "./layer/input.h"
#include "./layer/layer.h"
#include "./layer/loss_layer.h"

static void add_layer(std::shared_ptr<Layer> parent,
                      std::shared_ptr<Layer> child) {
    parent->addChild(child);
    child->addParent(parent);
}

NetDistributed::NetDistributed(std::shared_ptr<DistWorker> worker_,
                               std::shared_ptr<CudaContext> ctx_)
    : ctx(ctx_)
    , worker(worker_) {
}

NetDistributed::NetDistributed(std::shared_ptr<DistWorker> worker_,
                               std::shared_ptr<CudaContext> ctx_,
                               vector<std::shared_ptr<Layer>>& v_layers)
    : ctx(ctx_)
    , worker(worker_)
    , layers(v_layers) {
}

void NetDistributed::add(std::shared_ptr<Layer> l) {
    if (!layers.empty()) {
        ::add_layer(layers.back(), l);
    }
    layers.push_back(l);
}

void NetDistributed::add_layer(std::shared_ptr<Layer> l) {
    CHECK(!layers.empty()) << "There is no layer in the NetDistributedwork! "
                              "Please call "
                              "set_data() first.\n";
    this->add(l);
}

void NetDistributed::Init() {
    this->rank = ps::MyRank();
    this->InitCtx();
    this->params_size = 0;
    for (auto& l : layers) {
        l->Init();
        this->params_size += l->params_size();
    }
    this->loss->Init();
    this->InitWeights();
}

void NetDistributed::InitCtx() {
    for (auto& l : layers) {
        l->ctx = this->ctx;
    }
}

size_t NetDistributed::params() const {
    CHECK(params_size != 0) << "Please call NetDistributed::compile() first!";
    return params_size;
}

std::shared_ptr<FloatTensor> NetDistributed::forward(
    std::shared_ptr<FloatTensor> input) {
    data_input->FeedData(input);
    for (auto& l : layers) {
        l->Forward();
    }
    return layers.back()->get_output();
}

float NetDistributed::compute_loss(std::shared_ptr<FloatTensor> predict,
                                   std::shared_ptr<FloatTensor> label) {
    loss->loss_delta(predict, label);
    return loss->loss_value(predict, label);
}

void NetDistributed::backward() {
    for (int i = layers.size() - 1; i >= 0; i--) {
        layers[i]->Backward();
    }
}

void NetDistributed::updateWeights() {
    for (auto& l : layers) {
        l->updateWeights(this->lr);
    }
}

void NetDistributed::set_loss(const std::string& loss_name) {
    loss = std::make_shared<LossLayer>(loss_name, ctx);
    CHECK(!layers.empty()) << "There is no layer in the NetDistributedwork!\n";
    ::add_layer(layers.back(), loss);
}

void NetDistributed::set_data(vector<int>& shape) {
    CHECK(layers.empty()) << "The NetDistributedwork is not empty!";
    data_input = std::make_shared<DataInput>(shape);
    this->add(std::make_shared<Input>(data_input.get()));
}

void NetDistributed::set_lr(const float learning_rate) {
    this->lr = learning_rate;
}

void NetDistributed::print_layer() const {
    for (auto& l : layers) {
        std::cout << l->name << " ";
    }
    std::cout << std::endl;
}

void NetDistributed::compile(const std::string& loss_name,
                             std::shared_ptr<Metric> metric) {
    this->set_loss(loss_name);
    this->metric = metric;
    this->Init();
}

void NetDistributed::fit(const int epoch,
                         const float learning_rate,
                         BatchDataset* train_data,
                         BatchDataset* test_data) {
    // 初始化server的权重
    std::cout << "Network rank: " << this->rank << " start training"
              << std::endl;
    this->pushServerWeights();
    this->set_lr(learning_rate);
    this->train(epoch, train_data, test_data);
}

void NetDistributed::train(int epoch,
                           BatchDataset* train_data,
                           BatchDataset* test_data) {
    int iterations = train_data->batch_num;

    for (int e = 0; e < epoch; e++) {
        float batch_loss   = 0.f;
        float batch_metric = 0.f;
        train_data->reset();
        for (int i = 0; i < iterations; i++) {
            auto dp      = train_data->fetch();
            auto train_x = dp.first, train_y = dp.second;
            train_x->Reshape(data_input->output_shape);

            /* ****************************
                get server parameters
            **************************** */
            this->pullWeights();

            auto output = this->forward(train_x);

            float loss = this->compute_loss(output, train_y);
            batch_loss += loss;

            batch_metric += this->metric->value(output, train_y);

            this->backward();

            /* ****************************
                push local gradients
            **************************** */
            this->pushGradients();

            if (i % 50 == 0) {
                std::cout << GREEN << "******** Local Worker Running: " << rank
                          << " ********" << std::endl;
                std::cout << GREEN << "[" << i << "/" << iterations << "]"
                          << " iter loss: " << loss << std::endl;
            }
        }

        batch_loss /= iterations;
        batch_metric /= iterations;
        std::cout.precision(4);
        std::cout << GREEN << "=========== training phase: iterations["
                  << iterations << "]  "
                  << "batch[" << train_data->batch << "] ===========\n";
        std::cout << GREEN << "******** Local Worker Running: " << rank
                  << " ********" << std::endl;
        std::cout << "[" << e << "/" << epoch << "]  "
                  << "average loss: " << batch_loss << "  " << metric->name
                  << ": " << batch_metric << std::endl;

        if (test_data != nullptr) {
            this->pullWeights();
            this->test(test_data);
        }
    }
}

void NetDistributed::test(BatchDataset* test_data) {
    int iterations = test_data->batch_num;

    float test_loss   = 0.f;
    float test_metric = 0.f;
    test_data->reset();
    for (int i = 0; i < iterations; i++) {
        auto dp      = test_data->fetch();
        auto train_x = dp.first, train_y = dp.second;
        train_x->Reshape(data_input->output_shape);

        auto output = this->forward(train_x);

        float loss = this->compute_loss(output, train_y);
        test_loss += loss;

        test_metric += this->metric->value(output, train_y);
    }

    test_loss /= iterations;
    test_metric /= iterations;
    std::cout << YELLOW << "=========== test phase: iterations[" << iterations
              << "]  "
              << "batch[" << test_data->batch << "] ===========\n";
    std::cout << GREEN << "******** Local Worker Running: " << rank
              << " ********" << std::endl;
    std::cout.precision(4);
    std::cout << "test loss: " << test_loss << "  " << metric->name << ": "
              << test_metric << std::endl;
}

void NetDistributed::InitWeights() {
    for (auto& l : layers) {
        if (l->params.size()) {
            for (auto& p : l->params) {
                this->weights.push_back(p->toHost());
                this->layer_names2idxs[l->name].push_back(weights.size() - 1);
                this->keys.push_back(weights.size() - 1);
            }
        }
    }
}

void NetDistributed::getGradients() {
    this->gradients.clear();
    for (auto& l : layers) {
        if (l->params.size()) {
            for (auto& g : l->gradients) {  // gradients
                this->gradients.push_back(g->toHost());
            }
        }
    }
}

void NetDistributed::recoveryWeights() {
    for (auto& l : layers) {
        if (l->params.size()) {
            for (int i = 0; i < layer_names2idxs[l->name].size(); i++) {
                l->params[i]->fromHost(
                    this->weights[layer_names2idxs[l->name][i]]);
            }
        }
    }
}

void NetDistributed::pushServerWeights() {
    // Each worker will have a unique rank within [0, NumWorkers())
    if (rank == 0) {
        worker->push(this->keys, this->weights);
    } else {
        // wait for initing server weights
        sleep(5);
    }
}

void NetDistributed::pushGradients() {
    this->getGradients();
    worker->push(this->keys, this->gradients);
}

void NetDistributed::pullWeights() {
    worker->pull(this->keys, this->weights);
    this->recoveryWeights();
}