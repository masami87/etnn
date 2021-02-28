#include "initializer.h"

void IRandUniform::apply(FloatTensor *tensor) {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
    checkCurandErr(
        curandGenerateUniform(generator, tensor->data_ptr(), tensor->size));
}