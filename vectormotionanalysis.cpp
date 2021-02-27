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
#include <cmath>

using namespace cv;
using namespace std;

vectorMotionAnalysis::vectorMotionAnalysis(QString fileName)
{
    // vectorMotionAnalysisLucas(fileName.toUtf8().constData());
    // vectorMotionAnalysisFarneback(fileName);
    this->vectorMotionAnalysisFarneback(fileName);
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
    vector<vector<Point> > contours;
    vector<vector<Point> > poly;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours0;
    while(true) {
        Mat frame2, next;
        capture >> frame2;
        if (frame2.empty())
            break;
        cvtColor(frame2, next, COLOR_BGR2GRAY);
        Mat flow(prvs.size(), CV_32FC2);

        /*
        prev	first 8-bit single-channel input image.
        next	second input image of the same size and the same type as prev.
        flow	computed flow image that has the same size as prev and type CV_32FC2.

        pyr_scale parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous one.

        levels number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images
        winsize averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
        iterations	number of iterations the algorithm does at each pyramid level.
        poly_n	size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
        poly_sigma	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.

        flags	operation flags that can be a combination of the following:
        OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
        OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
        */

        calcOpticalFlowFarneback(prvs, next, flow, 0.8, 8, 20, 10, 5, 7, OPTFLOW_FARNEBACK_GAUSSIAN);
        // visualization.
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        // angle *= 180 / M_PI / 2.f;
        Mat _hsv[3], hsv, hsv8, bgr, capture, result;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);
        add(frame2, bgr, capture);
        imshow("frame2", capture);
        double scalar = cv::sum(magn_norm)[0];
        if (scalar > 10000) {
            cout << scalar << endl;
            cout << "Motion detected" << endl;
            // Is it fire?
            // magn_norm
            cout << "magn_norm" << magn_norm << endl;
            // detect with shape or something.
            imshow("fire", capture);
            cv::threshold(next, result, 147, 255, cv::THRESH_BINARY);
            imshow("Check fire", result);
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            contours.resize(contours.size());
            for (size_t i = 0; i < contours.size(); ++i) {
                Mat tmp;
                // You can try more different parameters
                approxPolyDP(contours[i], tmp, 3, true);
                poly.push_back(tmp);
            }

            Mat cnt_img = Mat::zeros( 500, 500, CV_8UC3);
            for (size_t i = 0; i < contours.size(); ++i) {
                drawContours(cnt_img, poly, i, {255, 255, 255}, 1, 8, hierarchy, 0);
            }
            imshow("contours fire", cnt_img);

        }
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        prvs = next;
    }
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



/*
    No need to go below just Backup code.
*/


/*

// Mat mask = Mat::zeros(next.size(), next.type());
// Norm of motion detection is higher than 12,000, maybe it is not fire.
// prev(y,x)∼next(y+flow(y,x)[1],x+flow(y,x)[0])
*/

// double scalar = cv::sum(magn_norm)[0];
// Norm of motion detection is higher than 12,000, maybe it is not fire.
// if (scalar > 12000) {
    // cout << magn_norm << endl;
//    cout << scalar << endl;
//    cout << "Motion detected" << endl;
    // Is it fire?
//}

/*
Computes a dense optical flow using the Gunnar Farneback's algorithm.
Parameters
prev	first 8-bit single-channel input image.
next	second input image of the same size and the same type as prev.
flow	computed flow image that has the same size as prev and type CV_32FC2.
pyr_scale	parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
levels	number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
winsize	averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
iterations	number of iterations the algorithm does at each pyramid level.
poly_n	size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_sigma	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
flags	operation flags that can be a combination of the following:
OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
The function finds an optical flow for each prev pixel using the [60] algorithm so that

prev(y,x)∼next(y+flow(y,x)[1],x+flow(y,x)[0])
Note
An example using the optical flow algorithm described by Gunnar Farneback can be found at opencv_source_code/samples/cpp/fback.cpp
(Python) An example using the optical flow algorithm described by Gunnar Farneback can be found at opencv_source_code/samples/python/opt_flow.py

*/

// _hsv[1] = 255;
// angle *= 180/(2 * acos(0.0))/2;
// cout << magn_norm << endl;
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
// calcOpticalFlowFarneback(prvs, next, flow, 0.5, 8, 100, 10, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN );
/*

# use 0 for webcam capturing
    # cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture('test/Pedestrian overpass.mp4')
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        # print(np.sum(mag[100:300, 100:300]))
        if (np.sum(mag)> 100000):
            print('motion detected')

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


*/
