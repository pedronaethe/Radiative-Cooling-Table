#include <cuda_runtime.h>
#include <device_launch_parameters.h>

texture<float, cudaTextureType4D> textureQ;

__global__ void fetchQKernel(float* result, int sizeP1, int sizeP2, int sizeP3, int sizeP4)
{
    int p1 = threadIdx.x;
    int p2 = blockIdx.x;
    int p3 = blockIdx.y;
    int p4 = blockIdx.z;

    float q = tex4D(textureQ, p1, p2, p3, p4);
    result[p1 + p2 * sizeP1 + p3 * sizeP1 * sizeP2 + p4 * sizeP1 * sizeP2 * sizeP3] = q;
}

int main()
{
    // Set the sizes of the parameters
    int sizeP1 = 10;
    int sizeP2 = 20;
    int sizeP3 = 30;
    int sizeP4 = 40;

    // Allocate memory for the result on the host
    int resultSize = sizeP1 * sizeP2 * sizeP3 * sizeP4;
    float* resultHost = (float*)malloc(resultSize * sizeof(float));

    // Allocate memory for the result on the device
    float* resultDevice;
    cudaMalloc((void**)&resultDevice, resultSize * sizeof(float));

    // Bind the texture to the device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTextureToArray(textureQ, resultDevice, channelDesc);

    // Set texture parameters
    textureQ.normalized = false;
    textureQ.filterMode = cudaFilterModePoint;
    textureQ.addressMode[0] = cudaAddressModeClamp;
    textureQ.addressMode[1] = cudaAddressModeClamp;
    textureQ.addressMode[2] = cudaAddressModeClamp;
    textureQ.addressMode[3] = cudaAddressModeClamp;

    // Launch the kernel
    dim3 blockSize(sizeP1, 1, 1);
    dim3 gridSize(sizeP2, sizeP3, sizeP4);
    fetchQKernel<<<gridSize, blockSize>>>(resultDevice, sizeP1, sizeP2, sizeP3, sizeP4);

    // Copy the result back to the host
    cudaMemcpy(resultHost, resultDevice, resultSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaUnbindTexture(textureQ);
    cudaFree(resultDevice);
    free(resultHost);

    return 0;
}
