#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ps/ps.h"

class PushReq {
 public:
    PushReq(const ps::KVMeta &req_meta_tmp,
            const ps::KVPairs<float> &req_data_tmp) {
        req_meta = req_meta_tmp;
        req_data = req_data_tmp;
    }

    ps::KVMeta req_meta;
    ps::KVPairs<float> req_data;
};

class SgdDistServer {
 private:
    ps::KVServer<float> *server_;

    std::unordered_map<ps::Key, std::vector<float>> weights_;

    int sync_mode_;

    float learning_rate_;

    std::vector<PushReq> push_req_;

    std::mutex mutex_;

 public:
    SgdDistServer(float lr, int sync)
        : learning_rate_(lr)
        , sync_mode_(sync)
        , server_(new ps::KVServer<float>(0)) {
        server_->set_request_handle(std::bind(&SgdDistServer::DataHandle,
                                              this,
                                              std::placeholders::_1,
                                              std::placeholders::_2,
                                              std::placeholders::_3));
    }

    ~SgdDistServer() {
        if (server_)
            delete server_;
        server_ = nullptr;
    }

    void DataHandle(const ps::KVMeta &req_meta,
                    const ps::KVPairs<float> &req_data,
                    ps::KVServer<float> *server) {
        auto n = req_data.keys.size();

        if (req_meta.push) {  // push请求
            if (weights_.empty()) {
                // 初始化
                size_t offset = 0;
                for (int i = 0; i < n; i++) {
                    auto key   = req_data.keys[i];
                    int length = req_data.lens[i];

                    std::vector<float> tmp(length);
                    for (int j = 0; j < length; j++) {
                        tmp[j] = req_data.vals[offset + j];
                    }
                    offset += length;
                    weights_[key] = std::move(tmp);
                }
                server->Response(req_meta);
                return;
            }

            // deal func
            auto sgd_function = [this](const ps::KVMeta &req_meta_tmp,
                                       const ps::KVPairs<float> &req_data_tmp) {
                int idx  = 0;
                size_t n = req_data_tmp.keys.size();
                for (size_t i = 0; i < n; ++i) {
                    auto &key     = req_data_tmp.keys[i];
                    auto &weights = weights_[key];
                    int length    = req_data_tmp.lens[i];
                    for (int j = 0; j < length; j++) {
                        // sgd
                        weights[j] -=
                            learning_rate_ * req_data_tmp.vals[idx + j];
                    }
                    idx += length;
                }
            };

            if (!sync_mode_) {  // 异步模式
                sgd_function(req_meta, req_data);
                server->Response(req_meta);
            } else {  // 同步模式
                mutex_.lock();
                if (push_req_.size() ==
                    ps::NumWorkers() - 1) {  // 当最后一个req到达时，即n -
                                             // 1个work的req都已经到达了
                    // 求梯度的均值
                    learning_rate_ = learning_rate_ / ps::NumWorkers();
                    // 处理最后一个到达的req
                    sgd_function(req_meta, req_data);
                    // 处理之前已经到达的req
                    for (auto &x : push_req_) {
                        sgd_function(x.req_meta, x.req_data);
                    }
                    server->Response(req_meta);
                    for (auto &x : push_req_) {
                        server->Response(x.req_meta);
                    }
                    push_req_.clear();
                } else {
                    // 继续等待还为到达的req
                    push_req_.push_back({req_meta, req_data});
                }
                mutex_.unlock();
            }
        } else {  // pull请求
            ps::KVPairs<float> response;
            response.keys = req_data.keys;
            int dim       = 0;
            for (size_t i = 0; i < n; ++i) {
                auto &key = req_data.keys[i];
                std::vector<float> weights(dim, 0.0);
                if (weights_.count(key)) {
                    weights = weights_[key];
                }
                dim = weights.size();
                response.lens.push_back(weights.size());
                for (auto x : weights) {
                    response.vals.push_back(x);
                }
            }
            server->Response(req_meta, response);
        }
    }
};