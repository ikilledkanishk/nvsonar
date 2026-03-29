/*
 * PCIe bandwidth benchmark
 *
 * Measures host-to-device and device-to-host transfer speed
 * using pinned memory for maximum throughput.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TRANSFER_SIZE (128 * 1024 * 1024)  // 128MB
#define ITERATIONS 10

typedef struct {
    double h2d_gbps;
    double d2h_gbps;
    int success;
    char error[256];
} BenchResult;

extern "C" void bench_pcie(BenchResult *result) {
    void *h_data = NULL, *d_data = NULL;
    cudaEvent_t start, stop;
    float elapsed;

    result->success = 0;
    result->h2d_gbps = 0;
    result->d2h_gbps = 0;

    size_t bytes = TRANSFER_SIZE;

    if (cudaMallocHost(&h_data, bytes) != cudaSuccess) {
        snprintf(result->error, 256, "failed to allocate pinned host memory");
        return;
    }

    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
        cudaFreeHost(h_data);
        snprintf(result->error, 256, "failed to allocate GPU memory");
        return;
    }

    memset(h_data, 0, bytes);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // host to device
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    result->h2d_gbps = ((double)bytes * ITERATIONS) / (elapsed / 1000.0) / 1e9;

    // device to host
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    result->d2h_gbps = ((double)bytes * ITERATIONS) / (elapsed / 1000.0) / 1e9;

    result->success = 1;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_data);
    cudaFree(d_data);
}
