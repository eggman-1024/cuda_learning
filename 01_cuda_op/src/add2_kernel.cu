#include "../include/add2.h"

__global__ void MatAdd(float* c, 
                      const float* a, 
                      const float* b, 
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * n + i;
    if (i < n && j < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_add2(float* c, 
                 const float* a, 
                 const float* b, 
                 int n) {
    dim3 block(16, 16); // 定义每个block有16×16=256个线程
    dim3 grid(n/block.x, n/block.y);    // 定义每个block有16×16=256个线程
    MatAdd<<<grid, block>>>(c, a, b, n);    // 启动CUDA kernel
}