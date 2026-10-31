#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main() {
    const size_t bytes = 1920UL * 1080UL * 3UL;   // one 1080p RGB frame, about 6.2 MB
    const int ITERS = 200;

    unsigned char* d_buf;                          // memory on the GPU
    CHECK(cudaMalloc(&d_buf, bytes));
    unsigned char* pageable = (unsigned char*)malloc(bytes);   // normal memory
    unsigned char* pinned;
    CHECK(cudaMallocHost(&pinned, bytes));         // pinned (page-locked) memory
    for (size_t i = 0; i < bytes; i++) pageable[i] = pinned[i] = (unsigned char)(i & 255);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    CHECK(cudaMemcpy(d_buf, pageable, bytes, cudaMemcpyHostToDevice));   // warm-up

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpy(d_buf, pageable, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("1) Pageable memcpy        : %.3f ms per frame\n", ms / ITERS);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpy(d_buf, pinned, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("2) Pinned memcpy          : %.3f ms per frame\n", ms / ITERS);

    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpyAsync(d_buf, pinned, bytes, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("3) Pinned + async stream  : %.3f ms per frame\n", ms / ITERS);

    cudaFreeHost(pinned); free(pageable); cudaFree(d_buf);
    cudaStreamDestroy(stream);
    return 0;
}
