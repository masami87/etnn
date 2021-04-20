#include "parser.h"

#include <iomanip>

#include "./layer/activation.h"
#include "./layer/conv2D.h"
#include "./layer/dense.h"
#include "./layer/softmax.h"
#include "./network.h"

enum class LayerType {
    DENSE,
    CONV2D,
    RELU,
    SIGMOID,
    SOFTMAX,
};

static const std::unordered_map<std::string, LayerType> TypeMap{
    {"dense", LayerType::DENSE},
    {"conv2d", LayerType::CONV2D},
    {"sigmoid", LayerType::SIGMOID},
    {"softmax", LayerType::SOFTMAX},
    {"relu", LayerType::RELU}};

bool parseDense(Net* net, typename nlohmann::basic_json<>::iterator& it);
bool parseConv2D(Net* net, typename nlohmann::basic_json<>::iterator& it);
bool parseSoftmax(Net* net, typename nlohmann::basic_json<>::iterator& it);
bool parseSigmoid(Net* net, typename nlohmann::basic_json<>::iterator& it);
bool parseReLu(Net* net, typename nlohmann::basic_json<>::iterator& it);

bool parseVector(json& j, std::vector<int>& vec);

void Load(json& j, Net* net, std::unordered_map<std::string, double>& args) {
    if (!parseDigits(j, args)) {
        throw std::runtime_error("cannot parse json args");
    }
    vector<int> tmp;
    if (!parseVector(j, tmp)) {
        throw std::runtime_error("cannot parse json shape");
    }
    net->set_data(tmp);

    if (!parseArch(j, net)) {
        throw std::runtime_error("cannot parse json arch filed");
    }
}

bool parseArch(json& j, Net* net) {
    auto archs = j["arch"];
    for (auto it = archs.begin(); it != archs.end(); ++it) {
        auto itype = it->find("type");
        if (itype == it->end()) {
            return false;
        }
        // auto type = TypeMap.at(itype->get<std::string>());
        auto type = TypeMap.find(itype->get<std::string>());
        if (type == TypeMap.end()) {
            return false;
        }
        auto t = type->second;
        switch (t) {
            case LayerType::DENSE:
                if (!parseDense(net, it)) {
                    return false;
                }
                break;
            case LayerType::CONV2D:
                if (!parseConv2D(net, it)) {
                    return false;
                }
                break;
            case LayerType::SIGMOID:
                if (!parseSigmoid(net, it)) {
                    return false;
                }
                break;
            case LayerType::RELU:
                if (!parseReLu(net, it)) {
                    return false;
                }
                break;
            case LayerType::SOFTMAX:
                if (!parseSoftmax(net, it)) {
                    return false;
                }
                break;
        }
    }
    return true;
}

bool parseDense(Net* net, typename nlohmann::basic_json<>::iterator& it) {
    auto idim  = it->find("dim");
    auto iname = it->find("name");
    if (idim == it->end() || iname == it->end()) {
        return false;
    }
    int dim   = idim->get<int>();
    auto name = iname->get<std::string>();
    net->add_layer(std::make_shared<Dense>(name, dim));

    return true;
}

bool parseSigmoid(Net* net, typename nlohmann::basic_json<>::iterator& it) {
    net->add_layer(std::make_shared<Activation>("sigmoid"));

    return true;
}

bool parseReLu(Net* net, typename nlohmann::basic_json<>::iterator& it) {
    net->add_layer(std::make_shared<Activation>("relu"));

    return true;
}

bool parseSoftmax(Net* net, typename nlohmann::basic_json<>::iterator& it) {
    net->add_layer(std::make_shared<Softmax>());

    return true;
}

bool parseConv2D(Net* net, typename nlohmann::basic_json<>::iterator& it) {
    //

    return false;
}

bool parseDigits(json& j, std::unordered_map<std::string, double>& args) {
    auto iepoch = j.find("epoch");
    auto ilr    = j.find("lr");
    auto ibatch = j.find("batch");
    if (iepoch == j.end() || ilr == j.end() || ibatch == j.end()) {
        return false;
    }
    args["epoch"] = iepoch->get<double>();
    args["lr"]    = ilr->get<double>();
    args["batch"] = ibatch->get<double>();
    return true;
}

bool parseVector(json& j, std::vector<int>& vec) {
    auto ishape = j.find("input_shape");
    if (ishape == j.end()) {
        return false;
    }
    for (auto it = ishape->begin(); it != ishape->end(); ++it) {
        auto n = it->get<int>();
        if (n <= 0)
            return false;
        vec.push_back(n);
    }
    return true;
}