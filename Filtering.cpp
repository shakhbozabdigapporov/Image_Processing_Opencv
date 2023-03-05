#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main()
{
    Mat image = imread("hw1_1.ppm", IMREAD_COLOR), convert_gray;
    cvtColor(image, convert_gray, COLOR_BGR2GRAY);
    Mat medianBlurImg;
    medianBlur(convert_gray, medianBlurImg, 5);// converted gray scale image used for Median filtering 
    // theshold values can be set to 5,6,7,9 etc.. We try to choose best one manually(I chose 5 as the best option)
    imshow("Original Image", image);
    imshow("Median Blurred Image", medianBlurImg);
    
    waitKey(0); // Wait for a keystroke in the window

    return 0;
}