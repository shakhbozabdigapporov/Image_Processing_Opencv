#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace cv;

int grab(VideoCapture cap, Mat frame);
int DisplayVideo(Mat frame, string windowName, string filterName);

CascadeClassifier face_cascade;
CascadeClassifier human_cascade;

int main() {

	VideoCapture cap(0);
	if (!cap.isOpened()) { cout << "not opened" << endl; return -1; }

	Mat frame;
	cap >> frame;

	face_cascade.load("C:/Program Files/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");
	human_cascade.load("C:/Program Files/Libraries/opencv/sources/data/haarcascades/haarcascade_fullbody.xml");

#pragma omp parallel sections
	{
#pragma omp section
		{
			printf("grab --> %d/n", omp_get_thread_num());
			grab(cap, frame);
		}
#pragma omp section
		{
			printf("Original Image --> %d/n", omp_get_thread_num());
			DisplayVideo(frame, "Original", "Original");

		}
#pragma omp section
		{
			printf("Thread for face --> %d/n", omp_get_thread_num());
			DisplayVideo(frame, "face", "face");
		}
#pragma omp section
		{
			printf("Thread for human detect --> %d/n", omp_get_thread_num());
			DisplayVideo(frame, "human", "human");
		}
	}
	printf("Total End/n");

	return 0;
}

int grab(VideoCapture cap, Mat frame) {
	while (1) {
		cap >> frame;
		if (!cap.isOpened()) { cout << "false" << endl; return -1; }
		if (waitKey(30) == 'q') break;
	}
	return 0;
}

int DisplayVideo(Mat frame, string windowName, string filter) {
	double fstart, fend, fprocTime;
	double fps;
	while (1) {
		fstart = omp_get_wtime();
		Mat result_frame = (frame).clone();

		if (frame.empty()) { cout << "empty" << endl; return -1; }

		if (filter == "face") {
			vector<Rect> faces;
			face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | 1, Size(60, 60));

			for (unsigned i = 0; i < faces.size(); i++) {
				rectangle(result_frame, faces[i], Scalar(0, 255, 0), 2, 2);
			}
		}
		if (filter == "human") {
			vector<Rect> human;
			human_cascade.detectMultiScale(frame, human, 1.1, 2, 0 | 1, Size(60, 60));

			for (unsigned i = 0; i < human.size(); i++) {
				rectangle(result_frame, human[i], Scalar(0, 0, 255), 2, 1);
			}
		}
		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		putText(result_frame, "FPS --> " + to_string(fps), Point(10, 25), FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2);

		namedWindow(windowName, 0);
		imshow(windowName, result_frame);
		if (waitKey(30) == 'q') break;

	}
	return 0;
}

