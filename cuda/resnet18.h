#ifndef RESNET18_H_INCLUDED
#define RESNET18_H_INCLUDED

#include <iostream>
using namespace std;

class Conv(){
    Conv(const int in_channel, const int out_channel, const int kernel_w, const int stride_w,
          const int pad_w, const int kernel_h = 0, const int stride_h = 0, const int pad_h = 0);
};

class MaxPool(){
    MaxPool(const int kernel_w, const int stride_w, const int pad_w, const int kernel_h = 0,
            const int stride_h = 0, const int pad_h = 0);
};

class GlobalAvgPool(){
    GlobalAvgPool();
};

class Gemm(){
    Gemm(const int in_channel, const int out_channel);
};

class TensorDiscriptor{
    float* tensor;
    int *shape;
    int shapelen;
    TensorDiscriptor(float* _tensor, int* _shape, int _shapelen): tensor(_tensor), shape(_shape), shapelen(_shapelen) {}
};
#endif // RESNET18_H_INCLUDED
