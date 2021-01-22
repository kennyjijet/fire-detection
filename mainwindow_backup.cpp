#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

cv::Mat	pixelMotion();
cv::Mat fireRGB(cv::Mat frame);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cv::VideoCapture video("test_fire_2.mp4");
        cv::Ptr<cv::BackgroundSubtractor> pMOG2;
        pMOG2 = cv::createBackgroundSubtractorMOG2();
        if (!video.isOpened()) {
            std::cout << "Can not open video" << std::endl;
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



            /*
            IplImage *redChannelImg;
            cv::Mat image1;
            IplImage* image2;
            image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
            // IplImage ipltemp;
            IplImage * ipl = ...;
            cv::Mat m = cv::cvarrToMat(ipl);
            */
            // cv::Mat frame;
            // IplImage temp = cvIplImage(channels[2]);
            // apply pre-processing functions
            // IplImage* frame2 = cvCloneImage(&IplImage(frame));


            // IplImage *redChannelImg;
            // redChannelImg = channels;
            // cv::cvarrToMat(redChannelImg);



            // cv::Mat image1;
            // IplImage* image2=cvCloneImage(&(IplImage)image1);

            // cvCopy(&redChannelImg, &(IplImage)image1);
            // redChannelImg = cvCloneImage(&(IplImage)channels[2]);
            // redChannelImg = &(IplImage)channels[2];
            // cvCopy(channels[2], redChannelImg);
            // redChannelImg = cvIplImage(channels[2]);
            // cvCopy(channels[2],redChannelImg);
            // redChannelImg = &IplImage(channels[2]);

            // cvCopy(channels[2], redChannelImg);

            // redChannelImg =
            // redChannelImg = cvIplImage(channels[2]);
            // redChannelImg = IplImage(channels[2]);
            // Mat mat_img(channels[2]);
            // IplImage* R = cvCreateImage(s, d, 1);
            // CvSize s = cvSize(src->width, src->height);
            // int d = src->depth;
            // redChannelImg = cvCreateImage(s, d, 1);
            // IplImage *redChannelImg;
            // &(IplImage)
            // IplImage *redChannelImg=cvCloneImage((IplImage*)channels[2]);
            // redChannelImg = cvIplImage(channels[2]);
/*

            cv::Mat image1;
            IplImage* image2;
            image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
            IplImage ipltemp;
            cvCopy(&ipltemp,image2);

*/
            IplImage copy = cvIplImage(channels[2]);
            IplImage* redChannelImg = &copy;
            // IplImage *redChannelImg;
            // cvCopy(&redChannelImg, channels[2]);
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

MainWindow::~MainWindow()
{
    delete ui;
}

cv::Mat pixelMotion()
{
    return cv::Mat();
}

cv::Mat fireRGB(cv::Mat frame)
{
    return cv::Mat();
}


