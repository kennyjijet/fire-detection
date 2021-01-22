#include "fire.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <QApplication>

fire::fire(QString path)
{
    // std::cout << "Current path is "  << '\n';
    // QDir directory("Documents/Letters");
    // QString path = directory.filePath("contents.txt");

    // std::cout << QDir::currentPath().toUtf8().constData()  << '\n';
    //std::string ss = QDir::currentPath().toUtf8().constData();
    //cv::VideoCapture video(ss);
    // Browsing from qt desktop.

    std::cout << path.toUtf8().constData() << '\n';
    // cv::VideoCapture video("D:/test_fire_2.mp4");

    cv::VideoCapture video(path.toUtf8().constData());
    cv::Ptr<cv::BackgroundSubtractor> pMOG2;
    pMOG2 = cv::createBackgroundSubtractorMOG2();
    if (!video.isOpened()) {
        std::cout << "Cannot open video" << std::endl;
        return;
    }

    cv::Mat originalFrame;
    cv::Mat resultFrame;

    cv::namedWindow("OriginalVideo", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("ResultVideo", cv::WINDOW_AUTOSIZE);

    std::vector<cv::Mat> channels;

    while (1) {

        video >> originalFrame;

        cv::medianBlur(originalFrame, resultFrame, 3);
        cv::split(resultFrame, channels);
        IplImage copy = cvIplImage(channels[2]);
        IplImage* redChannelImg = &copy;
        cvMaxS(redChannelImg, 150, redChannelImg);
        cv::Mat temp = cv::cvarrToMat(redChannelImg);
        pMOG2->apply(temp, temp);
        channels.clear();
        cv::imshow("OriginalVideo", originalFrame);
        cv::imshow("ResultVideo", temp);
        if (cv::waitKey(30) == 27) {
            break;
        }
    }
    cv::destroyAllWindows();
    originalFrame.release();
    resultFrame.release();
    pMOG2.release();
    video.release();
    return;
}
