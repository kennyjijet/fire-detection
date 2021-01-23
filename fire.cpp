#include "fire.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <QApplication>
#include <opencv2/core/mat.hpp>
#include <iostream>

fire::fire(QString path)
{
    cv::VideoCapture video(path.toUtf8().constData());
    if (!video.isOpened()) {
        std::cout << "Cannot open video" << std::endl;
        return;
    }
    cv::Mat originalFrame;
    cv::Mat blurFrame;
    cv::Mat hsvFrame;
    cv::Mat mask;
    cv::Mat output;
    int no_red = 0;
    std::vector<int> lower = {18, 50, 50};
    std::vector<int> upper = {35, 255, 255};
    cv::namedWindow("OriginalVideo", cv::WINDOW_AUTOSIZE);
    std::vector<cv::Mat> channels;

    while (true) {
        video >> originalFrame;
        cv::GaussianBlur(originalFrame, blurFrame, cv::Size(3, 3), 0); // Gaussian Blurring. In this method, instead of a box filter, a Gaussian kernel is used. It is done with the function
        cv::cvtColor(blurFrame, hsvFrame, cv::COLOR_BGR2HSV); // convert colour RGB/BGR to HSV
        cv::inRange(hsvFrame, lower, upper, mask); // Perform basic thresholding operations using OpenCV
        cv::bitwise_and(originalFrame,hsvFrame, output, mask); // Learn several arithmetic operations on images, like addition, subtraction, bitwise operations, and etc.
        no_red = cv::countNonZero(mask); // returns the number of nonzero pixels in the array matrix. Zero pixels denote black color and this function can come in handy if calculating the number of black pixels or number of white pixels
        if (no_red > 5000 && no_red < 10000) {
            std::cout << "Small Fire detected " << no_red << std::endl;
        } else if (no_red > 10000) {
            std::cout << "Big Fire detected " << no_red <<std::endl;
        } else {
            std::cout << "No Fire detected " << no_red <<std::endl;
        }
        channels.clear();
        cv::imshow("OriginalVideo", originalFrame);
        if (cv::waitKey(30) == 27) {
            break;
        }
    }
    cv::destroyAllWindows();
    video.release();
    return;
}
