#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ipp.h"

#include <opencv2/highgui.hpp>  
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;

extern "C" cudaError_t addWithCuda(int* c, const int* a, const int* b, size_t size);
extern "C" __global__ void addKernel(int* c, const int* a, const int* b);


extern "C" void gpu_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);

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


void Seq_tempGaussfilter(float Gvar, float Gpsi, int Gkernel_size, float* Gkernel)
{

	if (Gkernel_size % 2 == 0)
		Gkernel_size++;

	for (int x = -Gkernel_size / 2; x <= Gkernel_size / 2; x++) {
		for (int y = -Gkernel_size / 2; y <= Gkernel_size / 2; y++) {
			int index = (x + Gkernel_size / 2) * Gkernel_size + (y + Gkernel_size / 2);
			Gkernel[index] = exp(-((x * x) + (y * y)) / (2 * Gvar)) * (Gpsi * Gpsi);
		}
	}
}

void Seq_Gaussfilter(int Gkernel_size, float* Gausskernel)
{
	if (Gkernel_size == 5)
	{
		for (int i = 0; i < Gkernel_size * Gkernel_size; i++)
		{
			if (i == 0 || i == 4 || i == 20 || i == 24)
				Gausskernel[i] = float(float(1) / 273);
			else if (i == 1 || i == 3 || i == 5 || i == 9 || i == 15 || i == 19 || i == 21 || i == 23)
				Gausskernel[i] = float(float(4) / 273);
			else if (i == 2 || i == 10 || i == 14 || i == 22)
				Gausskernel[i] = float(float(7) / 273);
			else if (i == 6 || i == 8 || i == 16 || i == 18)
				Gausskernel[i] = float(float(16) / 273);
			else if (i == 7 || i == 11 || i == 13 || i == 17)
				Gausskernel[i] = float(float(26) / 273);
			else if (i == 12)
				Gausskernel[i] = float(float(41) / 273);
		}
	}
}

void GaussBlur_Serial(Mat pInput, Mat serial_result)
{
	float A[] = { 1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	7 / float(273),26 / float(273),41 / float(273), 26 / float(273), 7 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273) };
	Mat B(5, 5, CV_32F, A);

	int w = pInput.cols;
	int ws = pInput.cols;
	int h = pInput.rows;
	float Bsum = 0.0;
	for (int j = 2; j < h - 2; j++)
	{
		for (int i = 2; i < w - 2; i++)
		{
			Bsum = 0.0;
			for (int k = -2; k < 3; k++)
			{
				for (int r = -2; r < 3; r++)
				{
					Bsum += pInput.at<uchar>(j + k, i + r) * B.at<float>(k + 2, r + 2);
				}
			}
			serial_result.at<uchar>(j, i) = int(Bsum);
		}
	}
}

void GaussBlur_Omp(Mat pInput, Mat openmp_result)
{

	float A[] = { 1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	7 / float(273),26 / float(273),41 / float(273), 26 / float(273), 7 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273) };


	float C[] = { 1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	7 / float(273),26 / float(273),41 / float(273), 26 / float(273), 7 / float(273),
	4 / float(273),16 / float(273),26 / float(273), 16 / float(273), 4 / float(273),
	1 / float(273),4 / float(273),7 / float(273), 4 / float(273), 1 / float(273) };
	Mat D(5, 5, CV_32F, A);

	int w = pInput.cols;
	int ws = pInput.cols;
	int h = pInput.rows;
	float Dsum = 0.0;
	int j, i;

#pragma omp parallel
	{
#pragma omp sections private(Dsum, j, i)
		{
#pragma omp section
			{

				for (int j = 2; j < h / 2; j++)
				{
					for (int i = 2; i < w - 2; i++)
					{
						Dsum = 0.0;
						for (int k = -2; k < 3; k++)
						{
							for (int r = -2; r < 3; r++)
							{
								Dsum += pInput.at<uchar>(j + k, i + r) * D.at<float>(k + 2, r + 2);
							}
						}
						openmp_result.at<uchar>(j, i) = int(Dsum);
					}
				}

			}
#pragma omp section
			{
				for (int j = h / 2; j < h - 2; j++)
				{
					for (int i = 2; i < w - 2; i++)
					{
						Dsum = 0.0;
						for (int k = -2; k < 3; k++)
						{
							for (int r = -2; r < 3; r++)
							{
								Dsum += pInput.at<uchar>(j + k, i + r) * D.at<float>(k + 2, r + 2);
							}
						}
						openmp_result.at<uchar>(j, i) = int(Dsum);
					}
				}

			}
		}
	}

}

int main()
{
	int dev = 0;
	cudaError_t error_id = cudaGetDeviceCount(&dev);

	Mat pInput = imread("lena.jpg", IMREAD_GRAYSCALE); //load image using opencv
	namedWindow("Input", 0);
	imshow("Input", pInput);
	int w = pInput.cols;
	int ws = pInput.cols;
	int h = pInput.rows;

	float* pSrc = new float[w * h];
	float* pDst = new float[w * h];

	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++)
		{
			pSrc[j * w + i] = unsigned char(pInput.data[j * ws + i]);
		}
	}

	float tStart, tEnd;
	double process_time;
	tStart = cvGetTickCount();

	float* pcuSrc;
	float* pcuDst;
	float* pcuGkernel;
	// Allocate cuda device memory
	(cudaMalloc((void**)&pcuSrc, w * h * sizeof(float)));
	(cudaMalloc((void**)&pcuDst, w * h * sizeof(float)));
	(cudaMemcpy(pcuSrc, pSrc, w * h * sizeof(float), cudaMemcpyHostToDevice));

	int kernel_size = 5;
	float* Gkernel = new float[kernel_size * kernel_size];
	Seq_Gaborfilter(0.5, (180.0 * 3.141593 / 180), (0.55), (90 * 3.141593 / 180), kernel_size, Gkernel);

	float* Gausskernel = new float[kernel_size * kernel_size];
	Seq_Gaussfilter(kernel_size, Gausskernel);

	(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
	(cudaMemcpy(pcuGkernel, Gausskernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	gpu_Gabor(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size);

	(cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));

	tEnd = cvGetTickCount();// for check processing
	process_time = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec
	printf("\n\nProcess time CUDA --> %f msec\n\n\n", process_time);

	CvSize cvsize1 = { w ,h };

	Mat outputCuda = Mat(cvsize1, CV_8UC1, Scalar(0));
	Mat outputOpencv = Mat(cvsize1, CV_8UC1, Scalar(0));
	Mat outputIPP = Mat(cvsize1, CV_8UC1, Scalar(0));
	for (int y = 0; y < cvsize1.height; y++) {
		for (int x = 0; x < cvsize1.width; x++) {

			outputCuda.at<uchar>(y, x) = pDst[y * cvsize1.width + x];
		}
	}

	// Starting Opencv with bultin filtering function
	tStart = cvGetTickCount();
	GaussianBlur(pInput, outputOpencv, { 5,5 }, 0);
	tEnd = cvGetTickCount();// for check processing
	process_time = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec
	printf("\nOpenCV Process time --> %f msec\n", process_time);

	//Starting OMP Serial/ Parallel with using the written functions
	Mat outputSerialOmp = Mat(cvsize1, CV_8UC1, Scalar(0));
	Mat outputParallelOmp = Mat(cvsize1, CV_8UC1, Scalar(0));

	// Serial Gaussian Filter
	tStart = cvGetTickCount();
	GaussBlur_Serial(pInput, outputSerialOmp);
	tEnd = cvGetTickCount();// for check processing
	process_time = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec
	printf("\nSerialOMP Process Time --> %f msec\n", process_time);

	namedWindow("Serial_result", 0);
	imshow("Serial_result", outputSerialOmp);

	//Parallel OMP
	tStart = cvGetTickCount();
	GaussBlur_Omp(pInput, outputParallelOmp);
	tEnd = cvGetTickCount();// for check processing
	process_time = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec
	printf("\nOMP parallel Process time --> %f msec\n", process_time);
	namedWindow("openmp_result", 0);
	imshow("openmp_result", outputParallelOmp);

	tStart = cvGetTickCount();
	IppiSize size, tsize;

	size.width = w;
	size.height = h;

	Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* D_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* T_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);

	IppiMaskSize msize = (IppiMaskSize)55;
	int iSpecSize = 0, iTmpBufSize = 0;

	Ipp8u* pBuffer;
	IppFilterGaussianSpec* pSpec;
	ippiCopy_8u_C1R((const Ipp8u*)pInput.data, size.width, S_img, size.width, size);
	ippiFilterGaussianGetBufferSize(size, 5, ipp32f, 1, &iSpecSize, &iTmpBufSize);
	pSpec = (IppFilterGaussianSpec*)ippsMalloc_8u(iSpecSize);;
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	ippiFilterGaussianInit(size, 5, 1.f, ippBorderConst, ipp8u, 1, pSpec, pBuffer);
	ippiFilterGaussianBorder_8u_C1R(S_img, size.width, D_img, size.width, size, msize, pSpec, T_img);

	outputIPP = Mat(cvsize1, CV_8U, D_img);

	tEnd = cvGetTickCount();// for check processing
	process_time = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec
	printf("\nProcess time IPP --> %f msec\n", process_time);

	namedWindow("Cuda", 0);
	imshow("Cuda", outputCuda);

	namedWindow("opencv_result", 0);
	imshow("opencv_result", outputOpencv);

	namedWindow("ipp_result", 0);
	imshow("ipp_result", outputIPP);

	waitKey(0);
	// free the device memory

	cudaFree(pcuSrc);
	cudaFree(pcuDst);


	return 0;
}