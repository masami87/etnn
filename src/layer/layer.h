#pragma once
#include "../base.h"
#include "../device.h"
#include "../network.h"
#include "../tensor.h"
#include "../utils.h"

class Net;

class Layer {
 public:
    std::string name;

    vector<int> input_shape;
    vector<int> output_shape;

    Layer(const std::string &name, std::shared_ptr<CudaContext> &ctx_);

    Layer(const Layer &) = delete;
    Layer &operator=(const Layer &) = delete;
    virtual ~Layer();

    // For input layer
    virtual void Init()     = 0;
    virtual void Forward()  = 0;
    virtual void Backward() = 0;

    virtual void addParent(const std::shared_ptr<Layer> &l);
    virtual void addChild(const std::shared_ptr<Layer> &l);

    virtual void updateWeights(float lr);

    virtual std::shared_ptr<FloatTensor> get_output() const;
    virtual std::shared_ptr<FloatTensor> get_diff() const;

    size_t params_size() const;

    friend class Net;

 protected:
    std::shared_ptr<CudaContext> ctx;
    vector<std::shared_ptr<Layer>> parents;
    vector<std::shared_ptr<Layer>> childs;

    std::shared_ptr<FloatTensor> input;
    std::shared_ptr<FloatTensor> output;

    vector<std::shared_ptr<FloatTensor>> params;

    vector<std::shared_ptr<FloatTensor>> gradients;

    // diff to prev layer
    std::shared_ptr<FloatTensor> diff;
};