#include <ipp.h>
#include <opencv2/highgui.hpp>  
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>
using namespace cv;
using namespace std;

int Gaussian(int k)
{
	Mat scr, image;
	scr = imread("Grab_Image.bmp", 0);
	clock_t tStart = clock();
	resize(scr, image, Size(k, k));
	IppiSize size, tsize;
	size.width = image.cols;
	size.height = image.rows;
	Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* D_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* T_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	ippiCopy_8u_C1R((const Ipp8u*)image.data, size.width, S_img, size.width, size);
	tsize.width = image.cols;
	tsize.height = image.rows;
	Ipp8u KernelSize = 5;
	int pSpecSize = 0;
	int pBufferSize = 0;
	Ipp8u sigma = 3;
	IppFilterGaussianSpec* pSpec = (IppFilterGaussianSpec*)ippsMalloc_8u(pSpecSize);
	ippiFilterGaussianGetBufferSize(tsize, KernelSize, ipp8u, 1, &pSpecSize, &pBufferSize);
	pSpec = (IppFilterGaussianSpec*)ippsMalloc_8u(pSpecSize);
	ippiFilterGaussianInit(tsize, KernelSize, sigma, ippBorderConst, ipp8u, 1, pSpec, T_img);
	ippiFilterGaussianBorder_8u_C1R(S_img, size.width, D_img, size.width, tsize, ippBorderConst, pSpec, T_img);
	Size s;
	s.width = image.cols;
	s.height = image.rows; 

	Mat dst(s, CV_8U, (void*)D_img);
	imshow("IPP_Gaussian", dst);
	cout << "Gaussian function IPP time is --> " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
	waitKey(0);
	return 0;
}

int Median(int k)
{
	Mat scr, Image;
	scr = imread("Grab_Image.bmp", 0);
	clock_t tStart = clock();
	resize(scr, Image, Size(k, k));
	IppiSize size, tsize;
	size.width = Image.cols;
	size.height = Image.rows;
	Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* D_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp8u* T_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
	Ipp16u* T = (Ipp16u*)ippsMalloc_16u(size.width * size.height);
	ippiCopy_8u_C1R((const Ipp8u*)Image.data, size.width, S_img, size.width, size);
	tsize.width = Image.cols;
	tsize.height = Image.rows;
	IppiSize maskSize = { 5, 5 };
	ippiFilterMedianBorder_8u_C1R(S_img, size.width, D_img, size.width, tsize, maskSize, ippBorderConst, 255, T_img);
	Size s;
	s.width = Image.cols;
	s.height = Image.rows;

	Mat dst(s, CV_8U, D_img);
	imshow("IPP_Median", dst);
	cout << "Median function IPP time is --> " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
	waitKey(0);
	return 0;
}

int main()
{
	Mat scr;
	Mat image1, image2, image3, image4;
	Mat sizeGaus_1, sizeGaus_2, sizeGaus_3, sizeGaus_4;
	Mat sizeMedian_1, sizeMedian_2, sizeMedian_3, sizeMedian_4;
	scr = imread("Grab_Image.bmp", 0);
	int size1 = 256;
	int size2 = 512;
	int size3 = 1024;
	int size4 = 2048;

	resize(scr, image1, Size(size1, size1));
	resize(scr, image2, Size(size2, size2));
	resize(scr, image3, Size(size3, size3));
	resize(scr, image4, Size(size4, size4));

	Gaussian(256);
	clock_t tStart1 = clock();
	GaussianBlur(image1, sizeGaus_1, Size(5, 5), 0);
	cout << "Time Gaussian 1 -->" << (double)(clock() - tStart1) / CLOCKS_PER_SEC << endl;
	imshow("GaussianBlur 1 Opencv", sizeGaus_1);
	waitKey(0);

	Gaussian(512);
	clock_t tStart2 = clock();
	GaussianBlur(image1, sizeGaus_2, Size(5, 5), 0);
	cout << "Time Gaussian 2 -->" << (double)(clock() - tStart2) / CLOCKS_PER_SEC << endl;
	imshow("GaussianBlur 2 Opencv", sizeGaus_2);
	waitKey(0);

	Gaussian(1024);
	clock_t tStart3 = clock();
	GaussianBlur(image1, sizeGaus_3, Size(5, 5), 0);
	cout << "Time Gaussian 3 -->" << (double)(clock() - tStart3) / CLOCKS_PER_SEC << endl;
	imshow("GaussianBlur 3 Opencv", sizeGaus_3);
	waitKey(0);

	Gaussian(2048);
	clock_t tStart4 = clock();
	GaussianBlur(image1, sizeGaus_4, Size(5, 5), 0);
	cout << "Time Gaussian 4 -->" << (double)(clock() - tStart4) / CLOCKS_PER_SEC << endl;
	imshow("GaussianBlur 4 Opencv", sizeGaus_4);
	waitKey(0);

	Median(256);
	clock_t tStart5 = clock();
	medianBlur(image1, sizeMedian_1, 5);
	cout << "Time Median 1 -->" << (double)(clock() - tStart5) / CLOCKS_PER_SEC << endl;
	imshow("MedianBlur 1 Opencv", sizeMedian_1);
	waitKey(0);

	Median(512);
	clock_t tStart6 = clock();
	medianBlur(image2, sizeMedian_2, 5);
	cout << "Time Median 2 -->" << (double)(clock() - tStart6) / CLOCKS_PER_SEC << endl;
	imshow("MedianBlur 2 Opencv", sizeMedian_2);
	waitKey(0);

	Median(1024);
	clock_t tStart7 = clock();
	medianBlur(image3, sizeMedian_3, 5);
	cout << "Time Median 3 -->" << (double)(clock() - tStart7) / CLOCKS_PER_SEC << endl;
	imshow("MedianBlur 3 Opencv", sizeMedian_3);
	waitKey(0);

	Median(2048);
	clock_t tStart8 = clock();
	medianBlur(image4, sizeMedian_4, 5);
	cout << "Time Median 4 -->" << (double)(clock() - tStart8) / CLOCKS_PER_SEC << endl;
	imshow("MedianBlur 4 Opencv", sizeMedian_4);
	waitKey(0);
	return 0;
}