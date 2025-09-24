#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // Step 1: Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        shared[i] = val;
        if (val > max_val) max_val = val;
    }
    __syncthreads();

    // Step 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        shared[i] = expf(shared[i] - max_val);
        sum += shared[i];
    }
    __syncthreads();

    // Step 3: Write softmax output
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = shared[i] / sum;
    }
}

int main() {
    const int rows = 1024;
    const int cols = 512;
    size_t size = rows * cols * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel: one block per row, cols threads per block
    softmax_kernel<<<rows, min(cols, 1024), cols * sizeof(float)>>>(d_input, d_output, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Softmax kernel time: %f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Optionally print part of output
    printf("Output[0:10]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}