#include <opencv2/opencv.hpp>
#include "ncnn_centerface.h"
#include <chrono>

using namespace cv;
using namespace std;

#define RESIZE
#define SHOW_LANDMARKS

int main(int argc, char** argv) {
    float f;
    float FPS[16];
    size_t i, j;
    int Fcnt=0;
    Mat frame;
    FaceInfo* FacePtr;
	Centerface centerface;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

	centerface.init();

    VideoCapture cap("Walks2.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the video" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
    while(1){
//        frame=imread("selfie.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

        Tbegin = chrono::steady_clock::now();

        std::vector<FaceInfo> face_info;
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

#ifdef RESIZE
        centerface.detect(inmat, face_info, 320, 240);
#else
        centerface.detect(inmat, face_info, frame.cols, frame.rows);
#endif // RESIZE

        for(i=0; i<face_info.size(); i++){
            FacePtr=&face_info[i];
            cv::rectangle(frame, cv::Point(FacePtr->x1, FacePtr->y1), cv::Point(FacePtr->x2, FacePtr->y2), cv::Scalar(0, 255, 0), 2);
#ifdef SHOW_LANDMARKS
            for(j=0; j<5; j++){
                cv::circle(frame, cv::Point(FacePtr->landmarks[2*j], FacePtr->landmarks[2*j+1]), 2, Scalar(0, 255, 255), -1);
            }
#endif // SHOW_LANDMARKS
        }
        cv::imwrite("selfie_result.jpg", frame); //save the result

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, format("FPS %0.2f", f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

        //show output
        imshow("RPi 4 - 1,95 GHz - 2 Mb RAM", frame);

        char esc = waitKey(5);
        if(esc == 27) break;
    }

    cout << "Closing the camera" << endl;
    destroyAllWindows();
    cout << "Bye!" << endl;
    return 0;
}
