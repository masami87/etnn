# ETNN

Etnn is a tiny deep learning library written in C++ and CUDA. You can built a simple classifier neural network with this. It is built on cudnn and cublas for efficient operators. Thanks [XNET](https://github.com/lyx-x/XNet) for cudnn examples.


## Build
```
cd etnn/
mkdir build && cd build/
cmake ..
cmake --build .
```

## Example
Please download MNIST datasets first to data directory and run example:
```
./build/examples/mnist_mlp data/
```

and distributed example:
```
./local.sh 1 2 build/examples/mnist_mlp_distributed data
```