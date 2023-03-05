// detecting blobs with Hough detection 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	string src_array[4] = { "circle_blob1.png","circle_blob2.png","circle_blob3.png","circle_blob4.png" };
	for (int i = 0; i < 4; i++) {
		// Reading an image
		Mat src = imread(src_array[i], IMREAD_COLOR);
		vector<Vec3f> circles;
		imshow("original image", src);
		// Check if image is loaded fine
		if (src.empty()) {
			printf(" Error opening image\n");
			return EXIT_FAILURE;
		}
		else if(src_array[i]=="circle_blob1.png") {
			Mat gray;
			cvtColor(src, gray, COLOR_BGR2GRAY);
			medianBlur(gray, gray, 5);
			HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
				gray.rows / 16,  // change this value to detect circles with different distances to each other
				100, 40, 10, 30 // change the last two parameters for min_radius & max_radius to detect larger circles
			);
		}
		else if (src_array[i] == "circle_blob2.png") {
			Mat gray;
			cvtColor(src, gray, COLOR_BGR2GRAY);
			medianBlur(gray, gray, 5);
			HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
				gray.rows / 16,  // change this value to detect circles with different distances to each other
				100, 40, 10, 30 // change the last two parameters for min_radius & max_radius to detect larger circles
			);
		}
		else if (src_array[i] == "circle_blob3.png") {
			Mat gray;
			cvtColor(src, gray, COLOR_BGR2GRAY);
			medianBlur(gray, gray, 5);
			HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
				gray.rows / 16,  // change this value to detect circles with different distances to each other
				100, 20, 1, 30 // change the last two parameters for min_radius & max_radius to detect larger circles
			);
		}
		else if (src_array[i] == "circle_blob4.png") {
			Mat gray;
			cvtColor(src, gray, COLOR_BGR2GRAY);
			medianBlur(gray, gray, 5);
			HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
				gray.rows / 16,  // change this value to detect circles with different distances to each other
				100, 30, 10, 45 // change the last two parameters for min_radius & max_radius to detect larger circles
			);
		}
		for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// circling the center
			circle(src, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
			// circle outline
			int radius = c[2];
			circle(src, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
		}
		imshow("detected circles", src);
		waitKey();
	}

	return EXIT_SUCCESS;
}
// Detecting blobs with Blob detection(Old algorithm) did not perform well
/*#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>

using namespace cv;
using namespace std;
int main()
{

	// Read image
	//Mat circle = imread("circle_blob1.png");
	Mat circle = imread("circle_blob2.png",IMREAD_COLOR);
	//Mat circle = imread("circle_blob3.png");
	//Mat circle = imread("circle_blob4.png");
	if (!circle.empty())
	{
		vector<KeyPoint> kp;
		SimpleBlobDetector::Params param;

		param.filterByConvexity = true;
		param.maxConvexity = 0.87;

		param.filterByColor = true;
		param.minThreshold = 10;
		param.maxThreshold = 255;


		param.filterByArea = true;
		param.minArea = 1500;

		param.filterByInertia = true;
		param.minInertiaRatio = 0.01;

		param.filterByCircularity = true;
		param.minCircularity = 0.1;

		imshow("Blobs original", circle);
		Ptr <SimpleBlobDetector> detector = SimpleBlobDetector::create();

		detector->detect(circle, kp);
		Mat output;
		drawKeypoints(circle, kp, output, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("Blobs detected", output);
	}
	waitKey(0);
	destroyAllWindows();
	return 0;
}*/