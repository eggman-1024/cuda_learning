#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <chrono>  // 用于纳秒级计时

#define THREAD_PER_BLOCK 256

// 使用C++11的chrono库实现纳秒级计时
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds nanoseconds;

__device__ void warpReduce(volatile float* sdata, unsigned int tid){
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
}

// dim3 Grid( N/(2*THREAD_PER_BLOCK),1);
__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32) warpReduce(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

bool check(float* out, float* res, int n){
    for(int i = 0; i < n; i++){
        if(out[i] != res[i]){
            return false;
        }
    }
    return true;
}

int main(){
    const int N = 32 * 1024 * 1024; // 32M
    float* a = (float*)malloc(N * sizeof(float));
    float* d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int NUM_PER_BLOCK = THREAD_PER_BLOCK * 2; // 每个block处理两个元素
    int block_num = N / NUM_PER_BLOCK;
    float *out=(float *)malloc((block_num)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(block_num)*sizeof(float));
    float *res=(float *)malloc((block_num)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    // CPU计时开始（纳秒级）
    auto cpu_start = Clock::now();
    
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            cur+=a[i*NUM_PER_BLOCK+j];
        }
        res[i]=cur;
    }
    
    // CPU计时结束
    auto cpu_end = Clock::now();
    auto cpu_time_ns = std::chrono::duration_cast<nanoseconds>(cpu_end - cpu_start).count();
    double cpu_time_ms = cpu_time_ns / 1e6; // 转换为毫秒
    
    printf("CPU Reduction time: %.6f ms (%.0f ns)\n", cpu_time_ms, (double)cpu_time_ns);

    // 内存传输计时
    cudaEvent_t mem_start, mem_stop;
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);
    
    cudaEventRecord(mem_start);
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(mem_stop);
    cudaEventSynchronize(mem_stop);
    
    float mem_time_h2d = 0;
    cudaEventElapsedTime(&mem_time_h2d, mem_start, mem_stop);

    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);

    // GPU计时（使用CUDA事件，精度最高可达0.5微秒）
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热GPU，消除首次运行的额外开销
    reduce4<<<Grid,Block>>>(d_a,d_out);
    
    // 同步设备，确保预热已完成
    cudaDeviceSynchronize();
    
    // 计时开始
    cudaEventRecord(start);
    
    // 执行20次以获取更准确的平均时间
    for (int i = 0; i < 20; i++) {
        reduce4<<<Grid,Block>>>(d_a,d_out);
    }
    
    // 计时结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_kernel_time = 0;
    cudaEventElapsedTime(&gpu_kernel_time, start, stop);
    gpu_kernel_time /= 20.0f; // 计算平均时间
    
    // 数据传回计时
    cudaEventRecord(mem_start);
    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);
    cudaEventRecord(mem_stop);
    cudaEventSynchronize(mem_stop);
    
    float mem_time_d2h = 0;
    cudaEventElapsedTime(&mem_time_d2h, mem_start, mem_stop);

    // 计算性能指标
    float data_size_gb = (float)N * sizeof(float) / (1024*1024*1024); // 数据大小（GB）
    float kernel_bandwidth = data_size_gb / (gpu_kernel_time * 0.001); // GB/s
    float h2d_bandwidth = data_size_gb / (mem_time_h2d * 0.001); // GB/s
    float d2h_bandwidth = ((float)block_num * sizeof(float) / (1024*1024*1024)) / (mem_time_d2h * 0.001); // GB/s
    
    // 输出性能报告
    printf("\n===== Performance Report =====\n");
    printf("GPU kernel execution time: %.6f ms (%.0f ns)\n", gpu_kernel_time, gpu_kernel_time * 1e6);
    printf("Memory H2D transfer time: %.6f ms\n", mem_time_h2d);
    printf("Memory D2H transfer time: %.6f ms\n", mem_time_d2h);
    printf("Total GPU time: %.6f ms\n", gpu_kernel_time + mem_time_h2d + mem_time_d2h);
    printf("Speedup (kernel only): %.2fx\n", cpu_time_ms / gpu_kernel_time);
    printf("Speedup (including memory transfers): %.2fx\n", cpu_time_ms / (gpu_kernel_time + mem_time_h2d + mem_time_d2h));
    printf("Kernel Bandwidth: %.2f GB/s\n", kernel_bandwidth);
    printf("H2D Bandwidth: %.2f GB/s\n", h2d_bandwidth);
    printf("D2H Bandwidth: %.2f GB/s\n", d2h_bandwidth);
    printf("==============================\n\n");

    // 验证结果
    if(check(out,res,block_num)) {
        printf("Result verification: PASSED ✓\n");
    } else {
        printf("Result verification: FAILED ✗\n");
        printf("First few output values:\n");
        for(int i=0; i<std::min(10, block_num); i++){
            printf("out[%d]=%f, expected=%f\n", i, out[i], res[i]);
        }
    }

    // 释放资源
    free(a);
    free(out);
    free(res);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(mem_start);
    cudaEventDestroy(mem_stop);
    cudaFree(d_a);
    cudaFree(d_out);
    
    return 0;
}