/*
 * GPU compute throughput benchmark
 *
 * Saturates FP32 FMA units to measure peak compute throughput.
 * Each thread runs a chain of fused multiply-add operations.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE (64 * 1024 * 1024 / sizeof(float))  // 64MB
#define ITERATIONS 20
#define FMA_PER_THREAD 1024
#define BLOCK_SIZE 256

__global__ void fma_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = data[idx];
    float b = 1.00001f;
    float c = 0.99999f;

    // each FMA = 2 flops (multiply + add)
    for (int i = 0; i < FMA_PER_THREAD; i++) {
        a = a * b + c;
    }

    data[idx] = a;
}

typedef struct {
    double tflops;
    int success;
    char error[256];
} BenchResult;

extern "C" void bench_compute(BenchResult *result) {
    float *d_data = NULL;
    cudaEvent_t start, stop;
    float elapsed;

    result->success = 0;
    result->tflops = 0;

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);

    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
        snprintf(result->error, 256, "failed to allocate GPU memory");
        return;
    }
    cudaMemset(d_data, 0, bytes);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // warmup
    fma_kernel<<<grid, BLOCK_SIZE>>>(d_data, n);
    cudaDeviceSynchronize();

    // benchmark
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        fma_kernel<<<grid, BLOCK_SIZE>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    // 2 flops per FMA, FMA_PER_THREAD per thread, n threads, ITERATIONS runs
    double total_flops = 2.0 * FMA_PER_THREAD * (double)n * ITERATIONS;
    result->tflops = total_flops / (elapsed / 1000.0) / 1e12;

    result->success = 1;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
}
