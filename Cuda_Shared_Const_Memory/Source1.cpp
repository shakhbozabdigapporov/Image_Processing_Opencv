#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace cv;

extern "C" cudaError_t addWithCuda(int* c, const int* a, const int* b, size_t size);
extern "C" __global__ void addKernel(int* c, const int* a, const int* b);


extern "C" void gpu_const_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);
extern "C" void gpu_shared_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);

void Seq_Gaborfilter(float Gvar, float Gtheta, float Glambda, float Gpsi, int Gkernel_size, float* Gkernel)
{

	if (Gkernel_size % 2 == 0)
		Gkernel_size++;

	for (int x = -Gkernel_size / 2; x <= Gkernel_size / 2; x++) {
		for (int y = -Gkernel_size / 2; y <= Gkernel_size / 2; y++) {
			int index = (x + Gkernel_size / 2) * Gkernel_size + (y + Gkernel_size / 2);
			Gkernel[index] = exp(-((x * x) + (y * y)) / (2 * Gvar)) * cos(Glambda * (x * cos(Gtheta) + y * sin(Gtheta)) + Gpsi);
		}
	}
}

int main()
{
	Mat pInput = imread("Grab_Image.bmp", 0);
	namedWindow("input", 0);
	namedWindow("shared_output", 0);
	namedWindow("const_output", 0);
	imshow("input", pInput);
	int w = pInput.cols;
	int ws = pInput.cols;
	int h = pInput.rows;

	printf("\tInput Image Height: %d\n\tInput Image Width: %d\n", h, w);

	float* pDst = new float[w * h];
	Mat pfInput;
	pInput.convertTo(pfInput, CV_32FC1);

	clock_t before, now;


	float* pcuSrc;
	float* pcuDst;
	float* pcuGkernel;
	// Allocate cuda device memory
	(cudaMalloc((void**)&pcuSrc, w * h * sizeof(float)));
	(cudaMalloc((void**)&pcuDst, w * h * sizeof(float)));

	// copy input image across to the device
	(cudaMemcpy(pcuSrc, pfInput.data, w * h * sizeof(float), cudaMemcpyHostToDevice));

	//const int kernel_size = 5; // kernel size of 5
	const int kernel_size = 7; // kernel size of 7
	float* Gkernel = new float[kernel_size * kernel_size];
	Seq_Gaborfilter(0.5, (180.0 * 3.141593 / 180), (0.55), (90 * 3.141593 / 180), kernel_size, Gkernel);

	(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
	(cudaMemcpy(pcuGkernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
	{
		before = clock();
		(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
		(cudaMemcpy(pcuGkernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
		gpu_shared_Gabor(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size);
		// Copy the marker data back to the host
		(cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));

		now = clock();
		printf("Shared Memory Process time -->  %lf msec\n", (double)(now - before));
		Mat imgd1(Size(pInput.cols, pInput.rows), CV_32FC1, pDst);
		Mat dstdiplay;
		imgd1.convertTo(dstdiplay, CV_8UC1);
		imshow("shared_output", dstdiplay);
	}
	{
		before = clock();
		__constant__ float constKernel[kernel_size * kernel_size];
		cudaMemcpyToSymbol(constKernel, Gkernel, sizeof(float) * kernel_size * kernel_size);
		gpu_const_Gabor(pcuSrc, pcuDst, w, h, constKernel, kernel_size);
		// Copy the marker data back to the host
		(cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));

		now = clock();
		printf("Const Memory Process time --> %lf msec\n", (double)(now - before));
		Mat imgd1(Size(pInput.cols, pInput.rows), CV_32FC1, pDst);
		Mat dstdiplay;
		imgd1.convertTo(dstdiplay, CV_8UC1);
		imshow("const_output", dstdiplay);
	}


	waitKey(0);

	cudaFree(pcuSrc);
	cudaFree(pcuDst);

	return 0;
}
