#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

extern "C" void gpu_shared_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);
extern "C" void gpu_const_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);

__global__ void cuda_const_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, float* pKernel, int KWidth, int KHeight, float* pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int border;
	float temp;

	//	input[index] = clamp1(input);
	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		temp = 0;
		for (int i = 0; i < KHeight; i++) {
			for (int j = 0; j < KWidth; j++) {
				border = (y + j - KHeight / 2) * SrcWidth + (x + i);
				temp += pKernel[i * KWidth + j] * pSrcImage[border];
			}
		}
		// make cuda code here !!!!!!!!!!!!!!!!!!!!!!!!!
		pDstImage[index] = temp;

	}
	else
	{
		pDstImage[index] = 0;
	}
}

__global__ void cuda_shared_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, float* pKernel, int KWidth, int KHeight, float* pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int border;
	float temp;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	extern __shared__ float gmat[];
	if (tx < KWidth && ty < KHeight) {
		gmat[ty * KWidth + tx] = pKernel[ty * KWidth + tx];
	}

	__syncthreads();

	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		temp = 0;
		for (int i = 0; i < KHeight; i++) {
			for (int j = 0; j < KWidth; j++) {
				border = (y + j - KHeight / 2) * SrcWidth + (x + i);
				temp += gmat[i * KWidth + j] * pSrcImage[border];
			}
		}
		// make cuda code here !!!!!!!!!!!!!!!!!!!!!!!!!
		pDstImage[index] = temp;

	}
	else
	{
		pDstImage[index] = 0;
	}
}

void gpu_shared_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size)
{
	dim3 block = dim3(16, 16);
	dim3 grid = dim3(w / 16, h / 16);

	cuda_shared_Filter2D << < grid, block, sizeof(float)* kernel_size* kernel_size >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);

	cudaThreadSynchronize();
}

void gpu_const_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size)
{

	dim3 block = dim3(16, 16);
	dim3 grid = dim3(w / 16, h / 16);


	cuda_const_Filter2D << < grid, block >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);

	cudaThreadSynchronize();
}

