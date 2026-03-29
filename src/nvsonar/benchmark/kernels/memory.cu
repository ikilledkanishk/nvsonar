/*
 * GPU memory bandwidth benchmark
 *
 * Streams data through global memory to measure actual bandwidth.
 * Uses large arrays to bypass caches and measure raw HBM/GDDR speed.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE (256 * 1024 * 1024 / sizeof(float))  // 256MB
#define ITERATIONS 20
#define BLOCK_SIZE 256

__global__ void copy_kernel(float *dst, const float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void read_kernel(const float *src, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += src[i];
    }
    if (idx == 0) *result = sum;
}

__global__ void write_kernel(float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = 1.0f;
    }
}

typedef struct {
    double read_gbps;
    double write_gbps;
    double copy_gbps;
    int success;
    char error[256];
} BenchResult;

extern "C" void bench_memory(BenchResult *result) {
    float *d_src = NULL, *d_dst = NULL, *d_tmp = NULL;
    cudaEvent_t start, stop;
    float elapsed;

    result->success = 0;
    result->read_gbps = 0;
    result->write_gbps = 0;
    result->copy_gbps = 0;

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) {
        snprintf(result->error, 256, "failed to allocate GPU memory");
        return;
    }
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) {
        cudaFree(d_src);
        snprintf(result->error, 256, "failed to allocate GPU memory");
        return;
    }
    if (cudaMalloc(&d_tmp, sizeof(float)) != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        snprintf(result->error, 256, "failed to allocate GPU memory");
        return;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 65535) grid = 65535;

    // warmup
    copy_kernel<<<grid, BLOCK_SIZE>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();

    // read benchmark
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        read_kernel<<<grid, BLOCK_SIZE>>>(d_src, d_tmp, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    result->read_gbps = ((double)bytes * ITERATIONS) / (elapsed / 1000.0) / 1e9;

    // write benchmark
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        write_kernel<<<grid, BLOCK_SIZE>>>(d_dst, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    result->write_gbps = ((double)bytes * ITERATIONS) / (elapsed / 1000.0) / 1e9;

    // copy benchmark (read + write)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        copy_kernel<<<grid, BLOCK_SIZE>>>(d_dst, d_src, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    result->copy_gbps = ((double)bytes * 2 * ITERATIONS) / (elapsed / 1000.0) / 1e9;

    result->success = 1;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_tmp);
}
