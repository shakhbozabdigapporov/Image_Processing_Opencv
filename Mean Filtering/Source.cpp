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
void SSEmean_16bit(UINT16* src, int height, int width, UINT16* output)
{
	__m128i  temp, temp1, temp2;
	__m128i  row00;
	__m128i  row01;
	__m128i  row02;

	__m128i  row10;
	__m128i  row11;
	__m128i  row12;

	__m128i  row20;
	__m128i  row21;
	__m128i  row22;

	__m128i cof1 = _mm_set1_epi16(0);
	__m128 ps1, ps2, ps3, ps4;
	__m128i WtoDW0_0 = _mm_set1_epi16(0), WtoDW0_1 = _mm_set1_epi16(0);
	__m128i w1, w2, wLast;
	__m128 nine = _mm_set_ps1(9.0f);


	int nNewwidth = width / 8;
	for (int y = 0; y < height - 2; y++) {

		__m128i* dstLast = (__m128i*)output + y * (nNewwidth)+nNewwidth;

		for (int x = 0; x < nNewwidth; x++) {

			//////////////////////////////////////////////////
			row00 = _mm_load_si128((__m128i*)src + x + y * nNewwidth);
			row10 = _mm_load_si128((__m128i*)src + x + (y + 1) * nNewwidth);
			row20 = _mm_load_si128((__m128i*)src + x + (y + 2) * nNewwidth);

			temp = _mm_load_si128((__m128i*)src + x + y * nNewwidth + 1);

			//temp = (__m128i*)src + x + y * nNewwidth + 1;
			row01 = _mm_alignr_epi8(temp, row00, 2);
			row02 = _mm_alignr_epi8(temp, row00, 4);
			__m128i sumrow1 = _mm_add_epi16(_mm_add_epi16(row00, row01), row02);

			temp1 = _mm_load_si128((__m128i*)src + x + (y + 1) * nNewwidth + 1);
			row11 = _mm_alignr_epi8(temp1, row10, 2);
			row12 = _mm_alignr_epi8(temp1, row10, 4);
			__m128i sumrow2 = _mm_add_epi16(_mm_add_epi16(row10, row11), row12);

			temp2 = _mm_load_si128((__m128i*)src + x + (y + 2) * nNewwidth + 1);
			row21 = _mm_alignr_epi8(temp2, row20, 2);
			row22 = _mm_alignr_epi8(temp2, row20, 4);
			__m128i sumrow3 = _mm_add_epi16(_mm_add_epi16(row20, row21), row22);
			// add more code 

			__m128i sum3X3 = _mm_add_epi16(_mm_add_epi16(sumrow1, sumrow2), sumrow3);

			WtoDW0_0 = _mm_unpacklo_epi16(sum3X3, cof1);
			WtoDW0_1 = _mm_unpackhi_epi16(sum3X3, cof1);
			ps1 = _mm_cvtepi32_ps(WtoDW0_0);
			ps2 = _mm_cvtepi32_ps(WtoDW0_1);

			// divide scalar 9 : variable nine = _mm_set_ps1(9.0f);
			ps1 = _mm_div_ps(ps1, nine);
			ps2 = _mm_div_ps(ps2, nine);

			// type conversion to integer
			WtoDW0_0 = _mm_cvtps_epi32(ps1);
			WtoDW0_1 = _mm_cvtps_epi32(ps2);


			// Pack 2 register each having 4 elements to 8 elements in one register
			//w1 = _mm_packs_epi32(WtoDW0_0, WtoDW0_1);
			//w2 = _mm_packs_epi32(WtoDW1_0, WtoDW1_1);

			/////////////////////////////////////////////

					// Pack 8 elements to 16 elements 
			wLast = _mm_packus_epi16(WtoDW0_0, WtoDW0_1);

			// store the result
			_mm_store_si128(dstLast + x, wLast);

		}
	}
}

void SSEmean_8bit(Mat src_image, int height, int width)
{

	Mat output_final = Mat(height, width, CV_8UC1, Scalar(0));

	uchar* src = src_image.data;
	uchar* output = output_final.data;

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

	__m128i sum;

	__m128i row0_low, row0_high, row1_low, row1_high, row2_low, row2_high, row_add;
	__m128i row0_low_low, row0_low_high, row0_high_low, row0_high_high;

	__m128 coeff = _mm_set_ps1(9.0);
	__m128i zero = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);

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

			sum = _mm_packus_epi16(_mm_packs_epi32(row0_low_low, row0_low_high), _mm_packs_epi32(row0_high_low, row0_high_high));

			_mm_store_si128((__m128i*)output + (nNewwidth * (y)) + x, sum);
		}
	}
	//imshow("img_mean_8bit", output_final);
}

int main()
{
	Mat src_image = imread("lena.jpg");
	cvtColor(src_image, src_image, COLOR_BGR2GRAY);
	ushort* output = (ushort*)_mm_malloc(src_image.rows * src_image.cols * sizeof(ushort), 16);
	Mat src_image_16;// = Mat::zeros(src_image.rows, src_image.cols, CV_16UC1);;
	src_image.convertTo(src_image_16, CV_16UC1);

	int width = src_image.cols;
	int height = src_image.rows;
	clock_t before, now;


	before = clock();
	SSEmean_16bit((ushort*)src_image_16.data, src_image.rows, src_image.cols, output);
	now = clock();
	printf("16 bit Mean filter Processing Time--> %lf msec\n", (float)(now - before));


	before = clock();
	SSEmean_8bit(src_image, src_image.rows, src_image.cols);
	now = clock();
	printf("8 bit Mean filter Processing Time--> %lf msec\n", (float)(now - before));

	imshow("img_ori", src_image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
