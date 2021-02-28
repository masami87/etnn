#pragma once

#include "layer.h"
#include "../network.h"

class DataInput final {
public:
  vector<int> output_shape;
  DataInput(vector<int> &data_shape);

  DataInput(const DataInput &) = delete;

  DataInput &operator=(DataInput &) = delete;
  ~DataInput();

  void FeedData(std::shared_ptr<FloatTensor> &input);

  std::shared_ptr<FloatTensor> getData() const;

private:
  std::shared_ptr<FloatTensor> output;
};