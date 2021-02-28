#pragma once
#include "../base.h"
#include "../device.h"
#include <cstdio>
#include <iostream>

struct Op {
  std::string name;

  Op(const std::string &name, std::shared_ptr<CudaContext> &ctx_);

  Op &operator=(Op const &) = delete;
  Op(Op const &) = delete;

  virtual ~Op();

protected:
  std::shared_ptr<CudaContext> ctx;
};
