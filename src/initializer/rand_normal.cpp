#include "initializer.h"

void IRandNormal::apply(FloatTensor *tensor) {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
    curandGenerateNormal(
        generator, tensor->data_ptr(), tensor->size, mean, stddev);
    curandDestroyGenerator(generator);
}