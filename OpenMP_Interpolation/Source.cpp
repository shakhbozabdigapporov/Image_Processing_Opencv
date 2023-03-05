#include <tchar.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core_c.h>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

#define _nx 2
#define _ny 2
#define iteration 10

int wf = 4;

void wInter(int x, int y, string wf, float* w);
void Interp(unsigned char* src, int h, int width, string wf, float* w, int x, int y, unsigned char* output);
void Interp_omp(unsigned char* src, int h, int width, string wf, float* w, int x, int y, unsigned char* output);


using namespace std;

int main()
{
	int nx = _nx;
	int ny = _ny;
	int nsize = 2;
	//Mat src = imread("lena.jpg", 0);
	Mat src = imread("Grab_Image.bmp", 0);
	Size size(src.cols * nsize, src.rows * 2);
	resize(src, src, size, INTER_AREA);

	int wd = src.cols;
	int hg = src.rows;
	int width = src.cols;
	int height = src.rows;

	Mat dst(Size(wd * nx, hg * ny), CV_8UC1);

	int row, col;

	float* input = new float[wd * hg];
	memset(input, 0, wd * hg * sizeof(float));

	float* output = new float[(wd * nx) * (hg * ny)];
	memset(output, 0, (wd * nx) * (hg * ny) * sizeof(float));

	IplImage img = cvIplImage(src);

	for (row = 0; row < hg; row++)
		for (col = 0; col < wd; col++)
			input[row * wd + col] = (int)cvGetReal2D(&img, row, col);

	int iter = 0;
	int64 tStart, tEnd, tStart_1, tEnd_1;
	float pTime, pTime_1;
	float t_min = 0, t_max = 0;
	float t_ave = 0;

	float* w = new float[nx * 8];
	memset(w, 0, nx * 8 * sizeof(float));


	for (iter = 0; iter < iteration; iter++)
	{
		printf("\n%d 'th Iteration \n", iter + 1);
		wInter(nx, ny, "Bilinear", w);
		tStart = cvGetTickCount();
		Interp((unsigned char*)src.data, height, width, "Bilinear", w, nx, ny, (unsigned char*)dst.data);
		//Interp_omp((unsigned char*)src.data, height, width, "Bilinear", w, nx, ny, (unsigned char*)dst.data);
		tEnd = cvGetTickCount();
		pTime = 0.001 * (tEnd - tStart) / cvGetTickFrequency();
		t_ave += pTime;
		printf("Process Time_Inter %.3f ms\n", pTime);
		//printf("Process Time_Inter_OMP %.3f ms\n", pTime);
		if (iter == 0)
		{
			t_min = pTime;
			t_max = pTime;
		}
		else
		{
			if (pTime < t_min) t_min = pTime;
			if (pTime > t_max) t_max = pTime;
		}

	}

	if (iteration == 1) t_ave = t_ave;
	else if (iteration == 2) t_ave = t_ave / 2;
	else t_ave = (t_ave - t_min - t_max) / (iteration - 2);

	printf("\n\n Process Time on Average %.3f ms\n", t_ave);

	imshow("src", src);
	imshow("dst", dst);
	cvWaitKey(0);

	//imwrite("Lena_Result_1.jpg", dst);
	imwrite("Grab_Image_Result.bmp", dst);
	src.release();
	dst.release();
	delete[] input;
	delete[] output;

	return 0;
}

void wInter(int x, int y, string wf, float* w)
{
	x = x - 1;
	y = y - 1;
	int i;

	for (i = 0; i < x; i++)
	{
		w[i * 2 + 0] = 1 - (float)(i + 1) / (float)(x + 1);
		w[i * 2 + 1] = (float)(i + 1) / (float)(x + 1);
	}

}

void Interp(unsigned char* src, int hg, int wd, string wf, float* w, int x, int y, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int r, c, i, j, nc, nr, size;
	size = 1;

	int nwd = wd * (x + 1);

	float temp;

	for (r = 0; r < hg; r++) {
		for (c = 0 + size - 1; c < wd - size; c++)
		{
			nr = r * (y + 1);
			nc = c * (x + 1);

			output[nr * nwd + nc] = src[r * wd + c];

			for (i = 0; i < x; i++)
			{
				nc = c * (x + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
					temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];

				output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}


	int ntemp;

	for (r = 0 + size - 1; r < hg - size; r++) {
		for (c = 0 * (x + 1); c < wd * (x + 1) + x; c++)
		{
			for (i = 0; i < y; i++)
			{
				nr = r * (y + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
				{
					ntemp = (r - size + j + 1) * (y + 1);
					temp += w[i * (size * 2) + j] * (float)output[ntemp * nwd + c];
				}

				output[nr * nwd + c] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}
}

void Interp_omp(unsigned char* src, int hg, int wd, string wf, float* w, int x, int y, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int r, c, nc, nr, size;
	size = 1;

	int nwd = wd * (x + 1);

	int ntemp;
	float temp;

#pragma omp parallel sections private(temp, r, c, nc, nr)
	{
#pragma omp section
		{
			for (r = 0; r < hg; r++) {
				for (c = 0 + size - 1; c < (wd - size); c++)
				{
					nr = r * (y + 1);
					nc = c * (x + 1);

					output[nr * nwd + nc] = src[r * wd + c];

					for (int i = 0; i < x / 2; i++)
					{
						nc = c * (x + 1) + i + 1;
						temp = 0;
						for (int j = 0; j < size * 2; j++)
							temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];

						output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
					}
				}
			}
		}
#pragma omp section
		{
			for (r = 0; r < hg; r++) {
				for (c = 0 + size - 1; c < wd - size; c++)
				{
					nr = r * (y + 1);
					nc = c * (x + 1);

					output[nr * nwd + nc] = src[r * wd + c];

					for (int i = x / 2; i < x; i++)
					{
						nc = c * (x + 1) + i + 1;
						temp = 0;
						for (int j = 0; j < size * 2; j++)
							temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];

						output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
					}
				}
			}
		}
#pragma omp section
		{
			for (r = 0 + size - 1; r < hg - size; r++) {
				for (c = 0 * (x + 1); c < wd * (x + 1) + x; c++)
				{
					for (int i = 0; i < y; i++)
					{
						nr = r * (y + 1) + i + 1;
						temp = 0;
						for (int j = 0; j < size * 2; j++)
						{
							ntemp = (r - size + j + 1) * (y + 1);
							temp += w[i * (size * 2) + j] * (float)output[ntemp * nwd + c];
						}

						output[nr * nwd + c] = (unsigned char)((int)(temp + 0.5));
					}
				}
			}
		}
	}
}
