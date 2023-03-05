
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

extern "C" void gpu_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);

__global__ void cuda_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, float* pKernel, int KWidth, int KHeight, float* pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int border;
	float temp;

	//	input[index] = clamp1(input);
	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{

		// make cuda code here !!!!!!!!!!!!!!!!!!!!!!!!!
		pDstImage[index] = 255;

	}
	else
	{
		pDstImage[index] = 0;
	}


}



void gpu_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size)
{

	dim3 block = dim3(16, 16);
	dim3 grid = dim3(w / 16, h / 16);

	cuda_Filter2D <<< grid, block >>> (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);

	cudaThreadSynchronize();


	float* PrintKernel = new float[kernel_size * kernel_size];
	cudaMemcpy(PrintKernel, cuGkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++)
		{
			printf("%f\t", PrintKernel[i * kernel_size + j]);
		}
		printf("\n");
	}
}



