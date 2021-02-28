#include "op.h"
#include "device.h"

Op::Op(const std::string &name, std::shared_ptr<CudaContext> &ctx_)
    : name(name), ctx(ctx_) {}

Op::~Op() {}