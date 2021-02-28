#include "data_input.h"

DataInput::DataInput(vector<int> &data_shape) : output_shape(data_shape) {}

DataInput::~DataInput() {}

void DataInput::FeedData(std::shared_ptr<FloatTensor> &input) {
  CHECK(input->shape == output_shape);
  output = input;
}

std::shared_ptr<FloatTensor> DataInput::getData() const { return output; }