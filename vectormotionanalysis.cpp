#include "vectormotionanalysis.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <QApplication>
#include <opencv2/core/mat.hpp>
#include <QApplication>

using namespace cv;
using namespace std;

vectorMotionAnalysis::vectorMotionAnalysis(QString fileName)
{
    // vectorMotionAnalysisLucas(fileName.toUtf8().constData());
    // vectorMotionAnalysisFarneback(fileName);
    this->vectorMotionAnalysisFarneback(fileName);
}

void vectorMotionAnalysis::vectorMotionAnalysisLucas(QString fileName)
{
    // Lucas-Kanade: Sparse Optical Flow
    VideoCapture video(fileName.toUtf8().constData());
    if (!video.isOpened()){
        // error in opening the video input
        cerr << "Unable to open file!" << endl;
        return;
    }
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    // Take first frame and find corners in it.
    video >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes.
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    while(true) {
        Mat frame, frame_gray;
        video >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
        vector<Point2f> good_new;
        for(uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if(status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks for fire
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }

}

void vectorMotionAnalysis::vectorMotionAnalysisFarneback(QString fileName)
{
    // Dense Optical Flow
    VideoCapture capture(samples::findFile(fileName.toUtf8().constData()));
        if (!capture.isOpened()){
            //error in opening the video input
            cerr << "Unable to open file!" << endl;
            return;
        }
        Mat frame1, prvs;
        capture >> frame1;
        cvtColor(frame1, prvs, COLOR_BGR2GRAY);
        while(true) {
            Mat frame2, next;
            capture >> frame2;
            if (frame2.empty())
                break;
            cvtColor(frame2, next, COLOR_BGR2GRAY);
            Mat flow(prvs.size(), CV_32FC2);
            // prev(y,x)∼next(y+flow(y,x)[1],x+flow(y,x)[0])
            /*
cv2.calcOpticalFlowFarneback(im0, im1,
                    None, # flow
                    0.5, # pyr_scale
                    3, # levels
                    np.random.randint(3, 20), # winsize
                    3, #iterations
                    5, #poly_n
                    1.2, #poly_sigma
                    0 # flags
            */
            // calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 20, 3, 5, 1.2, 0);

            // calcOpticalFlowFarneback(prvs, next, flow, 0.5, 10, 20, 10, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
            calcOpticalFlowFarneback(prvs, next, flow, 0.5, 8, 100, 10, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN );
            // visualization.
            Mat flow_parts[2];
            split(flow, flow_parts);
            Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, COLOR_HSV2BGR);
            imshow("frame2", bgr);
            // imshow("Original", frame2);
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
            prvs = next;
        }

}
