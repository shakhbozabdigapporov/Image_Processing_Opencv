#include <tchar.h>
#include <smmintrin.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include <opencv2/highgui.hpp>  
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;


void SSE_mean(Mat src_image, int height, int width)
{
	int64 tStart, tEnd;
	float pTime;

	Mat clone = Mat(height, width, CV_8UC1, Scalar(0));

	uchar* src = src_image.data;
	uchar* output = clone.data;

	__m128i* temp1;
	__m128i* temp2;
	__m128i* temp3;

	__m128i* row00;
	__m128i  row01;
	__m128i  row02;

	__m128i* row10;
	__m128i  row11;
	__m128i  row12;

	__m128i* row20;
	__m128i  row21;
	__m128i  row22;

	__m128i sumrow1, sumrow2, sumrow3, asum;

	__m128i row0_low, row0_high, row1_low, row1_high, row2_low, row2_high, row_add;
	__m128i row0_low_low, row0_low_high, row0_high_low, row0_high_high;

	__m128 coeff = _mm_set_ps1(9.0);
	__m128i zero = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);

	tStart = cvGetTickCount();
	int nNewwidth = width / 16;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < nNewwidth; x++) {
			row00 = (__m128i*)src + x + y * (nNewwidth);
			row10 = (__m128i*)src + x + (y + 1) * (nNewwidth);
			row20 = (__m128i*)src + x + (y + 2) * (nNewwidth);

			temp1 = (__m128i*)src + x + y * (nNewwidth)+1;
			row01 = _mm_alignr_epi8(*temp1, *row00, 1);
			row02 = _mm_alignr_epi8(*temp1, *row00, 2);

			temp2 = (__m128i*)src + x + (y + 1) * (nNewwidth)+1;
			row11 = _mm_alignr_epi8(*temp2, *row10, 1);
			row12 = _mm_alignr_epi8(*temp2, *row10, 2);

			temp3 = (__m128i*)src + x + (y + 2) * (nNewwidth)+1;
			row21 = _mm_alignr_epi8(*temp3, *row20, 1);
			row22 = _mm_alignr_epi8(*temp3, *row20, 2);

			row0_low = _mm_unpacklo_epi8(*row00, zero);
			row_add = _mm_unpacklo_epi8(row01, zero);
			row0_low = _mm_add_epi16(row0_low, row_add);
			row_add = _mm_unpacklo_epi8(row02, zero);
			row0_low = _mm_add_epi16(row0_low, row_add);

			row0_high = _mm_unpackhi_epi8(*row00, zero);
			row_add = _mm_unpackhi_epi8(row01, zero);
			row0_high = _mm_add_epi16(row0_high, row_add);
			row_add = _mm_unpackhi_epi8(row02, zero);
			row0_high = _mm_add_epi16(row0_high, row_add);

			row1_low = _mm_unpacklo_epi8(*row10, zero);
			row_add = _mm_unpacklo_epi8(row11, zero);
			row1_low = _mm_add_epi16(row1_low, row_add);
			row_add = _mm_unpacklo_epi8(row12, zero);
			row1_low = _mm_add_epi16(row1_low, row_add);

			row1_high = _mm_unpackhi_epi8(*row10, zero);
			row_add = _mm_unpackhi_epi8(row11, zero);
			row1_high = _mm_add_epi16(row1_high, row_add);
			row_add = _mm_unpackhi_epi8(row12, zero);
			row1_high = _mm_add_epi16(row1_high, row_add);

			row2_low = _mm_unpacklo_epi8(*row20, zero);
			row_add = _mm_unpacklo_epi8(row21, zero);
			row2_low = _mm_add_epi16(row2_low, row_add);
			row_add = _mm_unpacklo_epi8(row22, zero);
			row2_low = _mm_add_epi16(row2_low, row_add);

			row2_high = _mm_unpackhi_epi8(*row20, zero);
			row_add = _mm_unpackhi_epi8(row21, zero);
			row2_high = _mm_add_epi16(row2_high, row_add);
			row_add = _mm_unpackhi_epi8(row22, zero);
			row2_high = _mm_add_epi16(row2_high, row_add);

			row0_low = _mm_add_epi16(_mm_add_epi16(row0_low, row1_low), row2_low);
			row0_high = _mm_add_epi16(_mm_add_epi16(row0_high, row1_high), row2_high);

			row0_low_low = _mm_unpacklo_epi16(row0_low, zero);
			row0_low_high = _mm_unpackhi_epi16(row0_low, zero);
			row0_high_low = _mm_unpacklo_epi16(row0_high, zero);
			row0_high_high = _mm_unpackhi_epi16(row0_high, zero);

			row0_low_low = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(row0_low_low), coeff));
			row0_low_high = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(row0_low_high), coeff));
			row0_high_low = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(row0_high_low), coeff));
			row0_high_high = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(row0_high_high), coeff));

			asum = _mm_packus_epi16(_mm_packs_epi32(row0_low_low, row0_low_high), _mm_packs_epi32(row0_high_low, row0_high_high));

			_mm_store_si128((__m128i*)output + (nNewwidth * (y)) + x, asum);
		}
	}

	tEnd = cvGetTickCount();
	pTime = 0.001 * (tEnd - tStart) / cvGetTickFrequency();
	printf("\nProcess Time Is --> %.3f ms\n", pTime);
	imshow("original image", src_image);
	imshow("Result image", clone);
	waitKey(0);
}

int main()
{
	int64 tStart, tEnd;
	float pTime;

	Mat src_image = imread("lena.jpg");
	cvtColor(src_image, src_image, COLOR_BGR2GRAY);

	int width = src_image.cols;
	int height = src_image.rows;

	SSE_mean(src_image, height, width);

	return 0;
}

